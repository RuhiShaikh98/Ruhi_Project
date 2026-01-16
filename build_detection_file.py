#!/use/bin/env python3

import csv
from pathlib import Path
from PIL import Image

#Paths
IMG_DIR = Path("/home/users/rshaikh/TEST_DETECTION/images")
PRED_DIR = Path("runs/predict_48h/final/labels")
OUT_CSV = Path("detection_48h.csv")

#Class mapping (matches spark.yaml)
ID_TO_CLASS = {
    0: "VenusExpress", 1: "Cheops", 2: "LisaPathfinder", 3: "ObservationSat1",
    4: "Proba2", 5:"Proba3", 6:"Proba3ocs", 7:"Smart1", 8:"Soho", 9:"XMM Newton"
}
DEFAULT_CLASS = "VenusExpress"

def yolo_to_xyxy(xc,yc,w,h,img_w,img_h):
    #Convert YOLO normalized (cx,cy,w,h) to pixel (xmin,ymin,xmax,ymax)
    xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
    xmin = max(0, int(round(xc - w/2)))
    ymin = max(0, int(round(yc - h/2)))
    xmax = min(img_w - 1, int(round(xc +w /2)))
    ymax = min(img_h - 1, int(round(yc + h/2)))
    return xmin,ymin,xmax,ymax

#Generate All 20,000 rows
rows = []
img_sizes = {} #Cache image dimensions

print("Processing 20,000 test images...")
for i, img_path in enumerate(sorted(IMG_DIR.glob("*.jpg")),1):
    filename = img_path.name
    base = img_path.stem
    txt_path = PRED_DIR / f"{base}.txt"

    #Get image size 
    if filename not in img_sizes:
        with Image.open(img_path) as im:
            img_sizes[filename] = im.size
    img_w, img_h = img_sizes[filename]

    #Try to read prediction .txt
    if txt_path.is_file():
        try:
            with txt_path.open() as f:
                lines = [line.strip().split() for line in f if line.strip()]
        except:
            lines = []
    else:
        lines = []

    if lines:
        #Take highest confidence prediction
        best_line = max(lines, key=lambda x: float(x[5]) if len(x) > 5 else 0)
        if len(best_line) >= 6:
            cls_id,xc,yc,bw,bh,conf = map(float, best_line[:6])
            xmin,ymin,xmax,ymax = yolo_to_xyxy(xc,yc,bh,img_w,img_h)
            cls_name = ID_TO_CLASS.get(int(cls_id), DEFAULT_CLASS)
            bbox_str = f"({xmin},{ymin},{xmax},{ymax})"
        else:
            cls_name = DEFAULT_CLASS
            bbox_str = "(0,0,0,0)"
    else:
        #No prediction -> dummy row
        cls_name = DEFAULT_CLASS
        bbox_str = "(0,0,0,0)"

    rows.append([filename,cls_name,bbox_str])

    if i % 5000 == 0:
        print(f"Processed {i}/20,000 images...")

#Write CSV
with OUT_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","class", "bbox"])
    writer.writerows(rows)

print(f"SUCCESS: Wrote {len(rows)} rows to {OUT_CSV}")
print(f"Predictions: {sum(1 for row in rows if row[2] != '(0,0,0,0)')} real, {20_000 - sum(1 for row in rows if row[2] != '(0,0,0,0)')} dummy")

