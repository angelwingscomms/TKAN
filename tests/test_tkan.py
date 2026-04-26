import copy

import numpy as np
import pandas as pd

from tkan import DEFAULTS, build_feature_frame, build_samples, compute_atr


def make_ohlc(rows=5000):
    index = pd.date_range('2026-01-01', periods=rows, freq='min')
    close = 100.0 + np.arange(rows) * 0.02 + np.sin(np.arange(rows) / 20.0) * 0.01
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.01
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.005
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=index)


def test_feature_builder_outputs_engineered_columns_only():
    df = make_ohlc()
    cfg = copy.deepcopy(DEFAULTS)
    cfg['symbol'] = 'BTCUSD'
    cfg['enabled_symbols'] = ['BTCUSD']

    features = build_feature_frame(df, cfg)

    assert 'open' not in features.columns
    assert 'close' not in features.columns
    assert 'BTCUSD_log_return_1' in features.columns
    assert 'BTCUSD_gk_vol_5' in features.columns
    assert 'BTCUSD_macd_hist_12_26_9' in features.columns
    assert 'BTCUSD_htf_15_rsi_14' in features.columns
    assert 'time_sin_hour' in features.columns
    assert len(features.dropna()) > 0


def test_feature_builder_and_labels_produce_samples():
    df = make_ohlc()
    cfg = copy.deepcopy(DEFAULTS)
    cfg['symbol'] = 'BTCUSD'
    cfg['enabled_symbols'] = ['BTCUSD']

    features = build_feature_frame(df, cfg)
    target = df.copy()
    merged = pd.concat([features, target.add_prefix('target_')], axis=1).dropna()
    features = merged[features.columns]
    target = merged[[f'target_{name}' for name in ('open', 'high', 'low', 'close')]]
    target.columns = ['open', 'high', 'low', 'close']
    atr = compute_atr(target, 9)

    X, y = build_samples(
        features,
        target,
        atr,
        sequence_length=20,
        horizon=5,
        tp_pct=0.02,
        tolerance=0.2,
        target_type='pct',
        atr_multiplier=1.0,
        tp_multiplier=1.0,
        use_hold=False,
    )

    assert X.ndim == 3
    assert X.shape[-1] == features.shape[1]
    assert len(X) == len(y)
    assert len(X) > 0


def test_build_samples_labels_hold_when_no_barrier_is_hit():
    features = pd.DataFrame({'feature': [0.0, 1.0, 2.0, 3.0]})
    target = pd.DataFrame({
        'open': [100.0, 100.0, 100.0, 100.0],
        'high': [100.1, 100.2, 100.3, 100.4],
        'low': [99.9, 99.8, 99.7, 99.6],
        'close': [100.0, 100.0, 100.0, 100.0],
    })
    atr = pd.Series([1.0, 1.0, 1.0, 1.0])

    X, y = build_samples(
        features,
        target,
        atr,
        sequence_length=2,
        horizon=2,
        tp_pct=1.0,
        tolerance=0.5,
        target_type='pct',
        atr_multiplier=1.0,
        tp_multiplier=1.0,
        use_hold=True,
    )

    assert X.shape[0] == 1
    assert y.shape == (1, 3)
    np.testing.assert_array_equal(y[0], np.array([0.0, 1.0, 0.0], dtype=np.float32))
