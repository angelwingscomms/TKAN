import shutil, sys
from pathlib import Path
src = Path('../../Files')
dst = Path('./data')
dst.mkdir(exist_ok=True)
shutil.copy(src / f'{sys.argv[1]}.csv', dst / f'{sys.argv[1]}.csv')