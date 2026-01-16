#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
from PIL import Image

PRED_DIR = Path("/home/users/rshaikh/FULL_SPARK2024/runs/predict_48h/final/labels")
IMG_DIR = Path("/home/users/rshaikh/TEST_DETECTION/images")
OUT_DIR = Path("/home/users/rshaikh/TEST_DETECTION/vis")

OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_TO_CLASS = {
    0: "VenusExpress",
    1: "Cheops",
    2: "LisaPathfinder",
    3: "ObservationSat1",
    4:  "Proba2",
    5: "Proba3",
    6: "Proba3ocs",
    7: "Smart1",
    8: "Soho",
    9: "XMM Newton"
}

def yolo_to_xyxy(xc,yc,w,h,img_w,img_h):
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    xmin = max(0,xc - w /2)
    ymin = max(0,yc - h /2)
    xmax = min(img_w -1, xc + w/2)
    ymax = min(img_h - 1,yc + h/2)
    return int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))

for txt_path in sorted(PRED_DIR.glob("*.txt")):
    image_base = txt_path.stem
    img_path = IMG_DIR / f"{image_base}.jpg"
    if not img_path.is_file():
        print(f"Missing image for {txt_path}")
        continue

    #Load image with OpenCV (BGR)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    #Image size (use PIL or OpenCV, both work)
    with Image.open(img_path) as im:
        w,h = im.size

    with txt_path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        xc,yc,bw,bh,conf = map(float,parts[1:])

        xmin,ymin,xmax,ymax = yolo_to_xyxy(xc,yc,bw,bh,w,h)
        cls_name = ID_TO_CLASS.get(cls_id, str(cls_id))

        #Draw rectangle and label
        color = (0,255,0)
        cv2.rectangle(img, (xmin,ymin),(xmax,ymax), color,2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img,label, (xmin,max(0,ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

    out_path = OUT_DIR / f"{image_base}.jpg"
    cv2.imwrite(str(out_path),img)
    print(f"Saved {out_path}")