#!/bin/bash
#SBATCH -p gpu                            #GPU partition
#SBATCH --gres=gpu:1                      #number of GPUs
#SBATCH -N 1                              #nodes
#SBATCH --cpus-per-task=16                #CPU cores
#SBATCH --mem=32G                         #RAM
#SBATCH --time=6:00:00                    #max wall time
#SBATCH --job-name=SPARK_Y11_PRED         #job name
#SBATCH --output=yolo11_predict48.log     #stdout/stderr log filename

#module load lang/Anaconda3/2020.11
#module load system/CUDA

#Activate environment with ultralytics installed
conda activate myenv

#Go to project root
cd /home/users/rshaikh/FULL_SPARK2024

#Run YOLO11 prediction on 80k test images
yolo detect predict \
    model=runs/detect_48h/final/weights/best.pt \
    source=/home/users/rshaikh/TEST_DETECTION/images \
    imgsz=640 \
    conf=0.25 \
    iou=0.45 \
    save_txt=True \
    save_conf=True \
    save=False \
    project=runs/predict_48h \
    name=final