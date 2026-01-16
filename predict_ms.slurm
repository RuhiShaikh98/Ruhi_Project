#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 --cpus-per-task=8 --mem=16G --time=3:00:00
#SBATCH -job-name=SPARK_MS
#SBATCH --output=ms.log

conda activate myenv
cd /home/users/rshaikh/FULL_SPARK2024

yolo detect predict\
    model=runs/detect_48h/final/weights/best.pt \
    source=/home/users/rshaikh/TEST_DETECTION/images \
    imgsz=736 conf=0.18 iou=0.42 save_txt=True save_conf=True save=False \
    project=runs/predict_ms name=ms736 \ 
    half=True
