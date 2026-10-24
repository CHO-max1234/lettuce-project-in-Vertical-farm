"""2. load reviewed bed numbers calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
import argparse
import json
from pathlib import Path

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    pointer = json.loads(CYCLE.OCR_POINTER.read_text(encoding='utf-8-sig'))
    manifest = Path(moved_path(pointer['manifest']))
    assert manifest.is_file(), str(manifest)
    ledger = moved_tree(json.loads(manifest.read_text(encoding='utf-8-sig')))
    output = CYCLE.PREPROCESS_OUTPUT / '2. OCR 기록'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='REVIEWED_OCR_INPUT_VALIDATED', source_pointer=str(CYCLE.OCR_POINTER), manifest=str(manifest), manifest_sha256=__import__('runtime').digest(manifest), human_decisions_recomputed=False, source_images_renamed=False)
    (output / 'reviewed_ocr_input.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
