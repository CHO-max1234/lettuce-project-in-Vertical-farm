"""Run scale estimation, streaming workbook export and value verification."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '3. 밝기보정과 Scale'
RUN_DIR = OUTPUT_ROOT / '_실행기록' / '261009_SCALE'
TEMP_WORKBOOK = OUTPUT_ROOT / '4작기_scale_map.writing.xlsx'
WORKBOOK = OUTPUT_ROOT / '4작기_scale_map.xlsx'
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args([] if 'ipykernel' in sys.modules else None)
    completion = RUN_DIR / 'completion.json'
    if completion.exists():
        prior = json.loads(completion.read_text(encoding='utf-8'))
        if prior['status'].startswith('COMPLETE') and WORKBOOK.exists() and (sha(WORKBOOK) == prior['verification']['workbook_sha256']):
            print('Already complete:', WORKBOOK)
            print(json.dumps(prior['status_counts'], ensure_ascii=False))
            return
    if not args.export_only:
        subprocess.run([sys.executable, '-u', str(CODE_DIR / 'scale_estimation.py')], check=True)
    if not (RUN_DIR / 'workbook_data.json').exists():
        raise FileNotFoundError('SCALE calculation data is missing')
    subprocess.run([sys.executable, '-u', str(CODE_DIR / 'scale_writer.py')], check=True)
    if WORKBOOK.exists() and sha(WORKBOOK) != sha(TEMP_WORKBOOK):
        raise FileExistsError('Different final workbook already exists: ' + str(WORKBOOK))
    if not WORKBOOK.exists():
        temp = WORKBOOK.with_suffix('.xlsx.tmp')
        shutil.copy2(TEMP_WORKBOOK, temp)
        os.replace(temp, WORKBOOK)
    subprocess.run([sys.executable, '-u', str(CODE_DIR / 'scale_validation.py')], check=True)
    print('Completed:', WORKBOOK)
if __name__ == '__main__':
    main()
