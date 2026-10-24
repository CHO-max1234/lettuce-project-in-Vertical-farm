"""Evaluate the 26 front-view targets with the standalone CNN separately."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUTPUT = CYCLE.MODEL_OUTPUT / '06_front_CNN'
import importlib.util
import sys
sys.dont_write_bytecode = True
spec = resolved_spec('s4_front_inputs', HERE / '2. Predict_stage.py')
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)
import hashlib, io, json, math, sqlite3, zlib, shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def main():
    protected = {str(p): s4.sha(p) for p in [s4.SOURCE115, s4.CNN_MODEL, Path(__file__), s4.REFERENCE / 'model/0. CNN Softlabel and merge.py', s4.REFERENCE / 'crop_integrity.py']}
    assert s4.sha(s4.CNN_MODEL) == s4.EXPECTED_CNN_SHA
    source = s4.read_sheet(s4.SOURCE115, '생육매칭')
    targets = [r for r in source if r['view'] == 'front']
    assert len(source) == 115 and len(targets) == 26
    assert all((t['camera'] == 'cam2' and t['A_C_main_view_eligible'] is False for t in targets))
    gate = s4.load('s4_front_gate', s4.REFERENCE / 'crop_integrity.py')
    pack = s4.load('s4_front_mask', s4.REFERENCE / 'preprocess/mask_record.py')
    assert not Path(str(s4.CROP_DB) + '-wal').exists() or Path(str(s4.CROP_DB) + '-wal').stat().st_size == 0
    snapshot = (s4.CROP_DB.stat().st_size, s4.CROP_DB.stat().st_mtime_ns)
    inputs = []
    with closing(sqlite3.connect(s4.CROP_DB.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        for t in targets:
            blob = db.execute('SELECT payload,mask FROM slots WHERE record_id=?', (t['record_id'],)).fetchone()
            assert blob and blob[1]
            r = json.loads(zlib.decompress(blob[0]))
            for k in ['record_id', 'bed_id', 'physical_slot', 'camera', 'crop_sha256']:
                assert str(r[k]) == str(t[k]), (t['item_id'], k)
            assert r['target_available'] is True
            mask = np.unpackbits(np.frombuffer(zlib.decompress(blob[1]), np.uint8), bitorder='little', count=math.prod(r['target_mask_shape'])).reshape(r['target_mask_shape']).astype(bool)
            assert hashlib.sha256(mask.tobytes()).hexdigest() == r['target_mask_pixels_sha256']
            assert np.array_equal(mask, pack.unpack(r) > 0)
            path = Path(moved_path(t['crop_path'])).resolve()
            assert path.is_relative_to(s4.CROP_ROOT.resolve())
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == r['crop_sha256']
            with Image.open(io.BytesIO(raw)) as im:
                rgb = np.asarray(im.convert('RGB'))
            assert rgb.shape[:2] == mask.shape and (not np.any(rgb[~mask]))
            assert hashlib.sha256(np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()).hexdigest() == r['crop_pixels_sha256']
            assert s4.IMAGE_NAME.fullmatch(path.name).groups() == tuple((str(r[k]) for k in ['bed_id', 'capture_date', 'capture_time', 'camera', 'physical_slot']))
            (allow, status, reason) = gate.judge(dict(r, has_crop='True', crop_path=str(path)), straight_ratio=0.8, minimum=20, main_share=0.9)
            row = {k: r[k] for k in ['record_id', 'bed_id', 'physical_slot', 'camera', 'capture_date', 'capture_time', 'crop_sha256']}
            row.update(crop_path=str(path), 분석가능=allow, crop_integrity_status=status, crop_integrity_reason=reason, prediction_scope='FRONT_ONLY_FROZEN_CNN_TRANSFER_DIAGNOSTIC')
            inputs.append(row)
    assert snapshot == (s4.CROP_DB.stat().st_size, s4.CROP_DB.stat().st_mtime_ns)
    mod = s4.load('s4_front_cnn', s4.REFERENCE / 'model/0. CNN Softlabel and merge.py')
    mod.IMAGE_NAME = s4.IMAGE_NAME
    mod.CROP_ROOT = s4.CROP_ROOT
    mod.MODEL_PT = s4.CNN_MODEL
    mod.DEVICE = __import__('runtime').torch_device()
    mod.BATCH_SIZE = 16
    mod.IMAGE_WORKERS = 4
    (model, transform, device, size, model_hash, signature) = mod.load_model()
    out = OUTPUT / ('run_' + s4.stamp())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / '실행코드_보존.py')
    headers = list(inputs[0])
    counter = Counter()
    with closing(sqlite3.connect(out / 'cnn_predictions_cache.sqlite3')) as db, ThreadPoolExecutor(max_workers=4) as pool:
        db.execute('CREATE TABLE predictions(signature TEXT,image_sha256 TEXT,probabilities TEXT,PRIMARY KEY(signature,image_sha256))')
        pred = mod.infer_chunk([(i + 2, [r[k] for k in headers]) for (i, r) in enumerate(inputs)], headers, model, transform, device, size, model_hash, signature, db, pool, counter)
    result = []
    for (t, r, p) in zip(targets, inputs, pred):
        result.append(dict(t, **r, **p))
    s4.write_csv(out / '01_label_free_front26_input.csv', inputs)
    s4.write_csv(out / '02_front26_predictions_and_truth.csv', result)
    good = [r for r in result if r['cnn_status'] == 'OK']
    y = [int(r['Stage_human']) for r in good]
    p = [int(r['cnn_pred_stage']) for r in good]
    metric = dict(target_n=26, predicted_n=len(good), unavailable_n=26 - len(good), accuracy=float(accuracy_score(y, p)) if y else None, macro_f1_012=float(f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0)) if y else None, confusion_truth0to4_pred0to4=confusion_matrix(y, p, labels=list(range(5))).tolist(), status_counts=dict(Counter((r['cnn_status'] for r in result))), gate_reason_counts=dict(Counter((r['crop_integrity_reason'] for r in result if r['crop_integrity_reason']))), scope='front transfer diagnostic only; not combined with top89 A/C performance', A_C_front_status='NOT_RUN_UNSUPPORTED_FRONT_VIEW_CONTRACT', model_training=False)
    for (path, digest) in protected.items():
        assert s4.sha(path) == digest
    s4.js(out / 'completion.json', dict(status='FRONT26_CNN_DIAGNOSTIC_COMPLETE', metrics=metric, source_hashes=protected, outputs={p.name: s4.sha(p) for p in out.iterdir() if p.is_file()}))
    s4.js(OUTPUT / 'latest_completed.json', dict(run=str(out)))
    print(json.dumps(dict(run=str(out), **metric), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
