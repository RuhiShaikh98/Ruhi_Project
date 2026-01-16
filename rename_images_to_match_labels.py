#!/use/bin/env python3

import os
from pathlib import Path

PROJECT_ROOT = Path(r"home\users\rshaikh\FULL_SPARK2024")
IMAGES_ROOT = PROJECT_ROOT/ "images"
LABELS_ROOT = PROJECT_ROOT/ "labels"

SPLITS = ["train", "val"]

def main():
    for split in SPLITS:
        label_dir = LABELS_ROOT / split
        if not label_dir.is_dir():
            print(f"Label dir missing: {label_dir}")
            continue
    
        print(f"\n=== Processing split: {split} ===")
        for label_path in sorted(label_dir.glob("*.txt")):
            label_name = label_path.stem
            #Split into class and original image base
            try:
                cls_name,img_base = label_name.split("_",1)
            except ValueError:
                print(f" [SKIP] Unexpected label name format: {label_name}")
                continue

            #Find the current JPG in the corresponding class/split folder
            img_dir = IMAGES_ROOT/cls_name/split
            if not img_dir.is_dir():
                print(f"[WARN] Image dir not found: {img_dir}")
                continue

            #Old image file is typically img_base + ".jpg"
            old_img_path = img_dir / f"{img_base}.jpg"
            if not old_img_path.is_file():
                #If not found try common variants
                candidates = list(img_dir.glob(f"*{img_base}*.jpg"))
                if len(candidates) == 1:
                    old_img_path = candidates[0]
                else:
                    print(f" [MISS] Could not find image for label {label_path.name} in {img_dir}")
                    continue

            new_img_path = img_dir / f"{label_name}.jpg"

            if new_img_path == old_img_path:
                #Already matches
                continue

            print(f"Renaming: {old_img_path.name} -> {new_img_path.name}")
            old_img_path.rename(new_img_path)
    
    print("\nDone renaming images.")

if __name__ == "__main__":
    main()