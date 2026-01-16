#!/bin/bash
#SBATCH -p gpu                            #GPU partition
#SBATCH --gres=gpu:1                      #number of GPUs
#SBATCH -N 1                              #nodes
#SBATCH --cpus-per-task=16                #CPU cores
#SBATCH --mem=32G                         #RAM
#SBATCH --time=48:00:00                   #max wall time
#SBATCH --job-name=SPARK_YOLO11_PRED      #job name
#SBATCH --output=val_yolo11.log           #stdout/stderr log filename

#module load lang/Anaconda3/2020.11
#module load system/CUDA

#Activate environment with ultralytics installed
conda activate myenv

#Go to project root
cd /home/users/rshaikh/FULL_SPARK2024

#python convert_csv_to_yolo.py

#Train YOLOv11 detection model 
yolo detect val \
    data=spark2024.yaml \
    model=yolo11l.pt \
    epochs=80 \
    imgsz=640 \
    batch=32 \
    device=0 