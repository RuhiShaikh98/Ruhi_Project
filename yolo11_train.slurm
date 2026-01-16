#!/bin/bash
#SBATCH -p gpu                            #GPU partition
#SBATCH --gres=gpu:1                      #number of GPUs
#SBATCH -N 1                              #nodes
#SBATCH --cpus-per-task=16                #CPU cores
#SBATCH --mem=32G                         #RAM
#SBATCH --time=48:00:00                   #max wall time
#SBATCH --job-name=SPARK_YOLO11_PRED      #job name
#SBATCH --output=train_yolo48.log         #stdout/stderr log filename

#module load lang/Anaconda3/2020.11
#module load system/CUDA

#Activate environment with ultralytics installed
conda activate myenv

#Go to project root
cd /home/users/rshaikh/FULL_SPARK2024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#python convert_csv_to_yolo.py
#Train YOLOv11 detection model
yolo detect train \
    data=spark2024.yaml \
    model=yolo11m.pt \
    epochs=80 \
    imgsz=640 \
    batch=24 \
    device=0 \
    lr0=0.01 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    close_mosaic=10 \
    patience=15 \
    save_period=20 \
    project=runs/detect_48h \
    name=final \