#!/usr/bin/env python3

import ast
import csv
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("/home/users/rshaikh/FULL_SPARK2024")

CSV_FILES = {
    "train": PROJECT_ROT / "train.csv",
    "val": PROJECT_ROOT / "val.csv",
}

IMAGES_ROOT = PROJECT_ROOT / "images"
LABELS_ROOT = PROJECT_ROOT / "labels"

CLASS_TO_ID = {
    "VenusExpress": 0,
    "Cheops": 1,
    "LisaPathfinder": 2,
    "ObservationSat1": 3,
    "Proba2": 4,
    "Proba3": 5,
    "Proba3ocs": 6,
    "Smart1": 7,
    "Soho": 8,
    "XMM Newton": 9,
}

def parse_bbox(bbox_str):
    #Parse bounding box column to (xmin,ymin,xmax,ymax)
    bbox_str = str(bbox_str).strip()
    if not (bbox_str.startswith("(")):
        bbox_str = f"({bbox_str})"
    x1,y1,x2,y2 = ast.literal_eval(bbox_str)
    return float(x1), float(y1), float(x2), float(y2)

def xyxy_to_yolo(xmin,ymin,xmax,ymax,img_w,img_h):
    cx  =((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h 
    bw = (xmax - xmin) / img_w 
    bh = (ymax - ymin) / img_h
    return cx,cy,bw,bh

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def process_split(split: str, csv_path: Path):
    assert split in {"train", "val"}
    print(f"\nProcessing {split} from {csv_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    out_dir = LABELS_ROOT / split
    ensure_dir(out_dir)

    img_dir = IMAGES_ROOT / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    img_size_cache = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            base_name = row["Image name"].strip()
            cls_name = row["Class"].strip()

            if cls_name not in CLASS_TO_ID:
                raise KeyError(f"Unknown class '{cls_name}' on line {i} of {csv_path}")
            
            class_id = CLASS_TO_ID[cls_name]
            xmin,ymin,xmax,ymax = parse_bbox(row["Bounding box"])

            #Actual image filename has class prefix:
            img_name = f"{cls_name}_{base_name}"
            img_path = img_dir / img_name
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found for row {i}: {img_path}")
            
            img_path_str = str(img_path.resolve())
            if img_path_str not in img_size_cache:
                with Image.open(img_path) as im:
                    w,h = im.size
                img_size_cache[img_path_str] = (w,h)
            else:
                w,h = img_size_cache[img_path_str]

            cx,cy,bw,bh = xyxy_to_yolo(xmin,ymin,xmax,ymax,w,h)

            #Label filename identical to image base name (without .jpg)
            base_no_ext = Path(img_name).stem
            label_filename = f"{base_no_ext}.txt"
            label_path = out_dir/label_filename

            line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            with label_path.open("a") as lf:
                lf.write(line)
            if i % 20000 == 0:
                print(f" processed {i} rows...")
    
    print(f"Done {split}. Labels written to {out_dir}")

def main():
    print("=== Converting SPARK CSV to YOLO labels ===")
    ensure_dir(LABELS_ROOT)
    for split,csv_path in CSV_FILES.items():
        process_split(split,csv_path)
    print("\nAll splits processed successfully.")

if __name__ == "__main__":
    main()

