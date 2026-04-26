import os
import shutil
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import argparse
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
import pandas as pd
from tkan import (
    load_config, load_csv, compute_atr, build_samples,
    normalize, save_norm_params, save_config, to_onnx_model, train, eval_loss, tkan_apply,
    build_feature_frame, select_symbol_ohlc, accuracy,
)

jax.default_backend = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, help='Config name (without extension) to load from ./config/')
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# TKAN TRAINING PIPELINE STARTING")
    print("#"*60 + "\n")

    cfg = load_config(args.config)
    seq_len = cfg['sequence_length']

    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    print(f"  Path: {cfg['data_path']}")
    print(f"  Parameters:")
    print(f"    - tp_pct: {cfg['threshold_pct']}")
    print(f"    - tolerance: {cfg['stop_loss_pct']}")
    print(f"    - horizon: {cfg['n_ahead']}")
    print(f"    - use_hold: {cfg['use_hold']}")
    print(f"    - target_type: {cfg['target_type']}")
    print(f"    - atr_multiplier: {cfg['atr_multiplier']}")
    print(f"    - tp_multiplier: {cfg['tp_multiplier']}")
    print(f"    - atr_period: {cfg['atr_period']}")
    print(f"    - target_symbol: {cfg['symbol']}")
    print(f"    - feature_symbols: {cfg['enabled_symbols']}")

    print(f"\n  Loading CSV file...")
    df = load_csv(f'./data/{cfg["data_path"]}')
    print(f"  CSV loaded: {len(df)} rows")

    features = build_feature_frame(df, cfg)
    target = select_symbol_ohlc(df, cfg['symbol'])
    merged = pd.concat([features, target.add_prefix('target_')], axis=1)
    valid_rows = merged.dropna()
    if len(valid_rows) == 0:
        raise ValueError(
            "No usable training rows were found. "
            "Your selected symbols and feature columns do not overlap at any timestamp, "
            "so every row is missing at least one value. "
            "Try enabling fewer symbols or using a data file where all selected symbols cover the same time range."
        )
    if len(valid_rows) < 2160:
        print(
            f"  Warning: only {len(valid_rows)} usable rows were found. "
            "This is very small, so training may be unstable."
        )
    merged = valid_rows
    features = merged[features.columns].copy()
    target = merged[[f'target_{field}' for field in ('open', 'high', 'low', 'close')]].copy()
    target.columns = ['open', 'high', 'low', 'close']
    print(f"  Feature data: {features.shape}")
    print(f"  Target OHLC: {target.shape}")

    print(f"\n  Computing ATR (period={cfg['atr_period']})...")
    atr = compute_atr(target, cfg['atr_period'])
    print(f"  ATR computed, first valid ATR at index: {atr.first_valid_index()}")

    print(f"\n  Building training samples (sequence_length={seq_len}, horizon={cfg['n_ahead']})...")
    X_arr, y_arr = build_samples(
        features, target, atr,
        sequence_length=seq_len,
        horizon=cfg['n_ahead'],
        tp_pct=cfg['threshold_pct'],
        tolerance=cfg['stop_loss_pct'],
        target_type=cfg['target_type'],
        atr_multiplier=cfg['atr_multiplier'],
        tp_multiplier=cfg['tp_multiplier'],
        use_hold=cfg['use_hold'],
    )

    total_candidates = max(0, len(features) - seq_len - cfg['n_ahead'] + 1)
    if cfg['use_hold']:
        buy_count = int(y_arr[:, 0].sum())
        hold_count = int(y_arr[:, 1].sum())
        sell_count = int(y_arr[:, 2].sum())
    else:
        buy_count = int(y_arr.sum())
        sell_count = len(y_arr) - buy_count
    dropped_count = total_candidates - len(X_arr)
    if len(X_arr) == 0:
        if cfg['use_hold']:
            raise ValueError("No labeled samples were produced. Too many windows were dropped as ambiguous.")
        raise ValueError("No clean labeled samples were produced. Too many windows were dropped as ambiguous or unresolved.")
    print(f"\n  Done! Generated {len(X_arr)} labeled samples")
    print(f"    - X shape: {X_arr.shape}")
    print(f"    - y shape: {y_arr.shape}")
    print(f"    - Buy labels:  {buy_count} ({100*buy_count/len(y_arr):.1f}%)")
    if cfg['use_hold']:
        print(f"    - Hold labels: {hold_count} ({100*hold_count/len(y_arr):.1f}%)")
    print(f"    - Sell labels: {sell_count} ({100*sell_count/len(y_arr):.1f}%)")
    print(f"    - Dropped windows: {dropped_count}")
    print("="*50 + "\n")

    print("\n" + "-"*50)
    print("SPLITTING DATA (TRAIN / VAL / TEST WITH PURGE GAP)")
    print("-"*50)
    split = cfg['train_test_split']
    gap = seq_len + cfg['n_ahead']
    if not 0 < split < 1:
        raise ValueError("train_test_split must be between 0 and 1.")
    usable = len(X_arr) - 2 * gap
    if usable <= 0:
        raise ValueError("Not enough samples for a purged train/val/test split. Add more data or reduce sequence_length/horizon.")
    train_n = int(usable * split)
    val_n = (usable - train_n) // 2
    test_n = usable - train_n - val_n
    val_start = train_n + gap
    test_start = val_start + val_n + gap
    X_tr, X_va, X_te = X_arr[:train_n], X_arr[val_start:val_start + val_n], X_arr[test_start:test_start + test_n]
    y_tr, y_va, y_te = y_arr[:train_n], y_arr[val_start:val_start + val_n], y_arr[test_start:test_start + test_n]
    if min(len(X_tr), len(X_va), len(X_te)) == 0:
        raise ValueError("Not enough samples for a purged train/val/test split. Add more data or reduce sequence_length/horizon.")
    print(f"  Training samples:   {len(X_tr)}")
    print(f"  Validation samples:{len(X_va)}")
    print(f"  Test samples:      {len(X_te)}")
    print(f"  Purge gap:         {gap}")

    print("\n" + "-"*50)
    print("NORMALIZING DATA")
    print("-"*50)
    xmin, xmax = X_tr.min(axis=(0, 1), keepdims=True), X_tr.max(axis=(0, 1), keepdims=True)
    print(f"  X_min range: [{xmin.min():.4f}, {xmin.max():.4f}]")
    print(f"  X_max range: [{xmax.min():.4f}, {xmax.max():.4f}]")
    X_tr, X_va, X_te, y_tr, y_va, y_te = normalize(xmin, xmax, X_tr, X_va, X_te, y_tr, y_va, y_te)
    print("  Normalization applied!")

    X_tr = jnp.array(X_tr)
    y_tr = jnp.array(y_tr)
    X_va = jnp.array(X_va)
    y_va = jnp.array(y_va)
    X_te = jnp.array(X_te)
    y_te = jnp.array(y_te)

    print(f"\n  Converted to JAX arrays:")
    print(f"    X_train: {X_tr.shape}")
    print(f"    y_train: {y_tr.shape}")
    print(f"    X_val:   {X_va.shape}")
    print(f"    y_val:   {y_va.shape}")
    print(f"    X_test:  {X_te.shape}")
    print(f"    y_test:  {y_te.shape}")
    print("-"*50 + "\n")

    hidden, sub = cfg['hidden_size'], cfg['sub_dim']
    input_dim = X_tr.shape[-1]
    output_dim = 3 if cfg['use_hold'] else 1

    print("\n=== TKAN ===")
    print(f"Architecture: {'dual-attention' if cfg['use_attention'] else 'base'}")
    params, train_losses, val_losses, train_accs, val_accs, elapsed = train(
        X_tr,
        y_tr,
        X_va,
        y_va,
        input_dim,
        hidden,
        sub,
        epochs=cfg['epochs'],
        lr=cfg['learning_rate'],
        batch_size=cfg['batch_size'],
        seed=cfg['seed'],
        output_dim=output_dim,
        use_attention=cfg['use_attention'],
        attn_dim=cfg['attention_dim'],
    )
    test_loss = float(eval_loss(params, X_te, y_te))
    test_preds = tkan_apply(params, X_te)
    test_acc = float(accuracy(test_preds, y_te))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Epoch':>5} | {'val_loss':>8} | {'val_acc':>8} | {'train_loss':>10} | {'train_acc':>10}")
    print("-"*60)
    for i, (vl, va, tl, ta) in enumerate(zip(val_losses, val_accs, train_losses, train_accs)):
        print(f"{i+1:>5} | {vl:>8.4f} | {va*100:>7.2f}% | {tl:>10.4f} | {ta*100:>9.2f}%")
    print("="*60)
    final_va = val_accs[-1]
    final_vl = val_losses[-1]
    print(f"Final val_loss: {final_vl:.4f} | val_acc: {100*final_va:.2f}% | Test loss: {test_loss:.4f} | Test acc: {100*test_acc:.2f}%")
    print(f"Total time: {elapsed:.1f}s")

    cfg['input_dim'] = int(input_dim)
    save_norm_params(xmin, xmax)
    save_config(cfg)

    print("\nExporting model to ONNX...")
    to_onnx_model(params, sequence_length=seq_len, input_dim=input_dim)
    print(f"Model saved to: model.onnx")
    model_path = Path('model.onnx')
    config_path = Path('config.mqh')
    norm_path = Path('norm_params.mqh')
    expert_path = Path('live.ex5')
    latest_input = max(path.stat().st_mtime for path in (model_path, config_path, norm_path))
    if not expert_path.exists() or expert_path.stat().st_mtime < latest_input:
        print("live.ex5 is older than model.onnx/config.mqh/norm_params.mqh. Recompile live.mq5 in MetaEditor before running the tester.")

    ts = datetime.now().strftime("%d%m-%H%M%S")
    version_dir = Path(f'models/{ts}')
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('model.onnx', version_dir / 'model.onnx')
    config_src = f'config/{args.config}.yaml' if args.config else 'config/default.yaml'
    shutil.copy(config_src, version_dir / 'config.yaml')
    shutil.copy('config.mqh', version_dir / 'config.mqh')
    shutil.copy('norm_params.mqh', version_dir / 'norm_params.mqh')
    (version_dir / 'model_version.txt').write_text(cfg['data_path'])
    print(f"Model version saved to: {version_dir}")

    best_val_loss = val_losses[jnp.argmin(jnp.array(val_losses)).item()]
    best_val_acc = val_accs[jnp.argmin(jnp.array(val_losses)).item()]
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))

    details = f"""# Training Details

**Total Parameters:** {param_count:,}
**Training Time:** {elapsed:.1f}s

## Epoch-by-Epoch Results

| Epoch | val_loss | val_acc | train_loss | train_acc |
|-------|----------|---------|------------|-----------|
"""
    for i, (vl, va, tl, ta) in enumerate(zip(val_losses, val_accs, train_losses, train_accs)):
        details += f"| {i+1} | {vl:.4f} | {va*100:.2f}% | {tl:.4f} | {ta*100:.2f}% |\n"

    details += f"""
## Final Metrics

- **Best Val Loss:** {best_val_loss:.4f}
- **Best Val Acc:** {100*best_val_acc:.2f}%
- **Final Val Loss:** {val_losses[-1]:.4f}
- **Final Val Acc:** {100*val_accs[-1]:.2f}%
- **Test Loss:** {test_loss:.4f}
- **Test Acc:** {100*test_acc:.2f}%
"""

    (version_dir / 'details.md').write_text(details)
    print(f"Training details saved to: {version_dir / 'details.md'}")

    live_mq5 = Path('live.mq5')
    if live_mq5.exists():
        content = live_mq5.read_text()
        content = content.replace('#include "config.mqh"', f'#include "models/{ts}/config.mqh"')
        content = content.replace('#resource "\\\\Experts\\\\TKAN\\\\model.onnx"', f'#resource "\\\\Experts\\\\TKAN\\\\models\\\\{ts}\\\\model.onnx"')
        live_mq5.write_text(content)
        print(f"Updated live.mq5 to use model: {ts}")


if __name__ == '__main__':
    main()
