"""Processing utilities for pipeline events."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from datetime import datetime
from pathlib import Path
import json
import re
import threading
import traceback
_root = None
_stage = 'standalone'
_lock = threading.Lock()

def configure(root):
    global _root
    _root = Path(root)
    _root.mkdir(parents=True, exist_ok=True)

def event(kind, message, **details):
    row = dict(time=datetime.now().astimezone().isoformat(), stage=_stage, kind=kind, message=message, **details)
    with _lock:
        print(f"[{row['time']}] [{_stage}] {message}", flush=True)
        if _root is not None:
            for name in ['events.jsonl', 'errors.jsonl'] if kind == 'ERROR' else ['events.jsonl']:
                with (_root / name).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

def start(stage):
    global _stage
    _stage = stage
    event('START', 'Starting stage.')

def report_error(source, error, trace=None):
    if trace is None and isinstance(error, BaseException):
        trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    locations = re.findall('File "([^"]+)", line (\\d+)', trace or '')
    (code_file, code_line) = locations[-1] if locations else ('', '')
    location = f' | Code: {code_file}:{code_line}' if code_file else ''
    event('ERROR', f'report error: | Input:{source}{location} | {error}', input_file=str(source), error=str(error), traceback=trace or '', code_file=code_file, code_line=code_line)
