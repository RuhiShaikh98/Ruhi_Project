#!/use/bin/env python3

#Ensemble : TTA + Original + Multi-scale

import csv
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

IMG_DIR = Path("/home/users/rshaikh/TEST_DETECTION/images")
PRED_DIRS = [
    "runs/predict_tta/tta/labels",
    "runs/predict_48h/final/labels",
    "runs/predict_ms/ms736/labels"
]
OUT_CSV = Path("detection_48h.csv")

#Class mapping
ID_TO_CLASS = {
    0: "VenusExpress", 1: "Cheops", 2:"LisaPathfinder", 3: "ObservationSat1",
    4: "Proba2", 5:"Proba3", 6: "Proba3ocs", 7: "Smart1", 8: "Soho", 9: "XMM Newton"
}
DEFAULT_CLASS = "VenusExpress"

def yolo_to_xyxy(xc,yc,w,h,img_w,img_h):
    xc *= img_w; yx *= img_h; w *= img_w; h *= img_h
    return (max(0, int(round(xc-w/2))), max(0,int(round(yc+h/2))))

print("ENSEMBLE START")
rows = []
img_sizes = {}

for i, img_path in enumerate(sorted(IMG_DIR.glob("*.jpg")),1):
    filename, base = img_path.name, img_path.stem

    #Image size cache
    if filename not in img_sizes:
        with Image.open(img_path) as im:
            img_sizes[filename] = im.size
    w,h = img_sizes[filename]

    #Collect predictions from all 3 runs
    all_preds = []
    for pred_dir in PRED_DIRS:
        txt_path = Path(pred_dir) / f"{base}.txt"
        if txt_path.is_file():
            with txt_path.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_id = int(parts[0])
                        cx,yc,bw,bh,conf = map(float,parts[1:6])
                        all_preds.append((cls_id,xc,yc,bw,bh,conf))
        
    if all_preds:
        #Group by class, sum confidence per class
        class_total_conf = defaultdict(float)
        class_weighted_coords = defaultdict(lambda: [0.0,0.0,0.0,0.0])

        for cls_id,xc,yc,bw,bh,conf in all_preds:
            class_weighted_coords[cls_id][0] += xc * conf #xc
            class_weighted_coords[cls_id][1] += yc * conf #yc
            class_weighted_coords[cls_id][2] += bw * conf #bw
            class_weighted_coords[cls_id][3] += bh * conf #bh

        #Best class = highest total confidence
        best_cls = max(class_total_conf, key=class_total_conf.get)
        total_conf = class_total_conf[best_cls]

        #Weighted average coordinates
        final_xc = class_weighted_coords[best_cls][0] / total_conf
        final_yc = class_weighted_coords[best_cls][1] / total_conf
        final_bw = class_weighted_coords[best_cls][2] / total_conf
        final_bh = class_weighted_coords[best_cls][3] / total_conf

        xmin,ymin,xmax,ymax = yolo_to_xyxy(final_xc,final_yc,final_bw,final_bh,w,h)
        cls_name = ID_TO_CLASS.get(best_cls,DEFAULT_CLASS)
        bbox_str = f"({xmin},{ymin},{xmax},{ymax})"
    else:
        cls_name,bbox_str = DEFAULT_CLASS,"(0,0,0,0)"
    
    rows.append([filename, cls_name, bbox_str])

    if i % 5000 == 0:
        print(f"Processed {i}/20,000 images...")

#Write FINAL CSV
with OUT_CSV.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename","class", "bbox"])
    writer.writerows(rows)

real_count = sum(1 for row in rows if row[2] != "(0,0,0,0)")
print(f"\n ENSEMBLE COMPLETE!")
print(f"{len(rows)} total rows -> {OUT_CSV}")
print(f"Real predictions: {real_count}, Dummies: {20000-real_count}")
print(f"UPLOAD detection_48h.csv")
