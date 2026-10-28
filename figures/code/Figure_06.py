"""Render five human-labelled heading-stage examples without model inference.

Fixed selections are from the development partition, with high-confidence,
one-hot human targets. They illustrate morphology, not classifier performance.
Run with --output-dir to generate the PNG/PDF and source audit together.
"""
from pathlib import Path
import argparse
import hashlib
import json
import csv

ROOT = Path(r'D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\1. 생육 단계 CNN-Soft label\5. CNN 재학습')
MANIFEST = ROOT / '00. 데이터와 분할' / 'manifest.csv'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'reproduced'
WIDTH_MM = 100
DPI = 600
# Populated from the source manifest; exact files, targets and hashes are checked.
SELECTIONS = [{'uid': 'legacy:bed30_bed30_20260320_082651_cam0_rf_b1_p01.png', 'sha256': '90c6eed5d8cfb947db069fceb3f4f4fea22249a0675a9ee24b872457b18708af'}, {'uid': 'legacy:bed39_bed39_20260414_093056_cam0_rf_t2_p01.png', 'sha256': '427b23629ca461bbaeeb11af165f5361a752a56f5799c21ca6c7105ecd9f0246'}, {'uid': 'legacy:bed35_bed35_20260415_092522_cam0_rf_t2_p01.png', 'sha256': '4e0e0b3d412e4c5a80ccf86a6da416d6335e29998245f180325ecf8b36284041'}, {'uid': 'legacy:bed95_bed95_20260410_113210_cam0_rf_b2_p01.png', 'sha256': 'a27cef5de9ac0bca9b5bfad3c17c588bf5c6346c1f6d7dd495e93ad5f5fba72b'}, {'uid': 'legacy:bed13_20260427_111122_cam0_rf_b3_p01.png', 'sha256': 'a28da1bf2f11106feef75d7e06189794e248846db950ab538262bf1030c999f2'}]

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_examples():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['uid']: row for row in csv.DictReader(stream)}
    examples = []
    for stage, selection in enumerate(SELECTIONS):
        row = rows[selection['uid']]
        assert int(row['stage']) == stage
        assert row['split'] == 'development' and row['confidence'] == 'high'
        assert float(row[f'soft_{stage}']) == 1.0
        path = Path(row['path'])
        if not path.exists():
            path = ROOT.parent / '3. 추가라벨링 1000장' / '600장' / row['image_name']
        assert digest(path) == selection['sha256'] == row['sha256']
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        examples.append({'stage': stage, 'panel': chr(65+stage), 'uid': row['uid'],
                         'path': str(path), 'sha256': row['sha256'], 'split': row['split'],
                         'confidence': row['confidence'], 'confidence_semantics': row['confidence_semantics'],
                         'origin': row['origin'], 'soft_target': [float(row[f'soft_{s}']) for s in range(5)],
                         'native_size': [pixels.shape[1], pixels.shape[0]], 'pixels': pixels})
    return examples

def render(examples, output_dir):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42})
    width_in = WIDTH_MM / 25.4
    fig = plt.figure(figsize=(width_in, width_in * .274), facecolor='white')
    photo_width = .184
    for i, example in enumerate(examples):
        left = .008 + i * .198
        ax = fig.add_axes([left, .245, photo_width, photo_width/.274], facecolor='black')
        # Preserve the whole source crop and its aspect ratio; black letterboxing only.
        pixels = example['pixels']; h, w = pixels.shape[:2]; side = max(w, h)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = pixels
        ax.imshow(square, interpolation='none'); ax.set_axis_off()
        fig.text(left+photo_width/2, .112, f'({example["panel"]})', ha='center', va='center', fontsize=8, fontweight='bold')
        example['effective_native_ppi_at_100mm'] = side / (photo_width * width_in)
    for extension in ['png', 'pdf']:
        fig.savefig(output_dir/f'Figure 6.{extension}', dpi=DPI, facecolor='white')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'Figure 6.png').exists():
        raise FileExistsError('Choose another output directory')
    source_sha = digest(MANIFEST)
    examples = load_examples()
    render(examples, args.output_dir)
    assert digest(MANIFEST) == source_sha
    report = {'status': 'PASS', 'manifest': str(MANIFEST), 'manifest_sha256': source_sha,
              'width_mm': WIDTH_MM, 'dpi': DPI, 'stage_order': [0,1,2,3,4],
              'selection_basis': 'Human high-confidence one-hot development examples; visual resemblance to the prior illustration. No classifier correctness filter.',
              'examples': [{k:v for k,v in e.items() if k != 'pixels'} for e in examples]}
    (args.output_dir/'Figure 6.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'figure': str(args.output_dir/'Figure 6.png'), 'native_ppi': [round(e['effective_native_ppi_at_100mm']) for e in examples]}))

if __name__ == '__main__':
    main()
