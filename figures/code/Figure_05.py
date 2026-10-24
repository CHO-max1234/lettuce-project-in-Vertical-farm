"""Reassemble Figure 5 using the four embedded photographs in its original PPTX.

This recovers the slide's image order, square placement, dates and arrows.
The PPTX does not identify the plant, bed, slot, camera or original crop filenames.
It therefore does not independently establish that all four photos are one plant.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import posixpath
import zipfile
import xml.etree.ElementTree as ET

INPUT = Path(r'D:\Googledrive\양상추 분류모델\관련논문\논문 구성\논문그림.pptx')
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'reproduced'
DPI = 600
FIGURE_WIDTH_IN = 7.2
NS = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_photos(path):
    with zipfile.ZipFile(path) as archive:
        root=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
        rels={r.attrib['Id']:posixpath.normpath('ppt/slides/'+r.attrib['Target'])
              for r in ET.fromstring(archive.read('ppt/slides/_rels/slide1.xml.rels'))}
        photos=[]
        for pic in root.findall('.//p:pic',NS):
            transform=pic.find('p:spPr/a:xfrm',NS)
            x=int(transform.find('a:off',NS).attrib['x'])
            y=int(transform.find('a:off',NS).attrib['y'])
            media=rels[pic.find('p:blipFill/a:blip',NS).attrib['{'+NS['r']+'}embed']]
            raw=archive.read(media)
            with Image.open(io.BytesIO(raw)) as image: pixels=np.asarray(image.convert('RGB'))
            photos.append(dict(x=x,y=y,media=media,pixels=pixels,native_size=[pixels.shape[1],pixels.shape[0]],sha256=hashlib.sha256(raw).hexdigest()))
        dates=[n.text.strip() for n in root.findall('.//a:t',NS) if n.text and n.text.strip()]
    photos.sort(key=lambda p:p['x'])
    assert len(photos)==4 and dates==['260314','260401','260420','260504']
    return photos,dates

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    args=parser.parse_args()
    digest=sha(args.input)
    photos,dates=load_photos(args.input)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/'Figure 5.png'
    if output.exists():raise FileExistsError('Choose another output directory')
    plt.rcParams.update({'font.family':'Arial','font.size':10,'figure.facecolor':'white'})
    fig=plt.figure(figsize=(FIGURE_WIDTH_IN,1.96))
    for i,(photo,date) in enumerate(zip(photos,dates)):
        left=.008+i*.253
        ax=fig.add_axes([left,.035,.226,.77])
        # The original slide places each photo in a square; preserve that layout.
        ax.imshow(photo['pixels'],aspect='auto',interpolation='none');ax.axis('off')
        fig.text(left+.113,.91,date,ha='center',va='center',fontsize=10)
        if i<3:
            x=left+.226;y=.40;w=.027;h=.105
            points=[(x,y-h/4),(x+w*.55,y-h/4),(x+w*.55,y-h/2),(x+w,y),(x+w*.55,y+h/2),(x+w*.55,y+h/4),(x,y+h/4)]
            fig.add_artist(Polygon(points,closed=True,transform=fig.transFigure,facecolor='white',edgecolor='#34758D',linewidth=.8))
    fig.savefig(output,dpi=DPI)
    plt.close(fig)
    assert sha(args.input)==digest
    report=dict(status='COMPOSITION_REPRODUCED_IDENTITY_UNVERIFIED',source_pptx=str(args.input),source_sha256=digest,
        output_sha256=sha(output),dates=dates,photos=[{k:v for k,v in p.items() if k!='pixels'} for p in photos],
        plant_identity_verified=False,original_crop_paths_verified=False,pixel_identical_to_collected_figure=False,
        layout='Original image order and square placement; newly rendered date labels and outlined arrows; original PNG preserved',
        dpi=DPI,native_detail_limited=True)
    (args.output_dir/'Figure 5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='photos'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
