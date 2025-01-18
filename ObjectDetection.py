import os
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import accuracy_score
import numpy as np
import json
import random
import shutil
from ultralytics.models.yolo.detect import DetectionTrainer
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


class ObjectDetector:
    def __init__(self,data,model):
        self.data = data
        self.model=model

    def train_yolo_model(self,epochs=50,patience=10,batch=16,lr0=0.001, imgsz=640,optimizer='auto'):
        model = YOLO(self.model)
        model.train(
            data=self.data,
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            optimizer=optimizer,
            project='./runs/detect',
        )
    def test_predict(self,model_path,test_dir,output_dir,num_images=10):
        model= YOLO(model_path)
        os.makedirs(output_dir,exist_ok=True)
        num_images = min(num_images), len(os.listdir(test_dir))
        test_images=[os.path.join(test_dir,f) for f in random.sample(os.listdir(test_dir),num_images)]
        model.predict(source=test_images,save=True,project=output_dir)

    def predict_with_yolo(self,test_images_dir,all_labels,model_path,imgsz=640):
        model=YOLO(model_path)
        results = model.predict(source=test_images_dir,imgsz=imgsz,stream=True)
        predictions = []
        for r in results:
            filename = os.path.basename(r.path)
            boxes= r.boxes.data.cpu().numpy()
            for box in boxes:
                cls=int(box[5])
                bbox=box[:4].tolist()
                predictions.append({'filename':filename,'class':all_labels[cls],'bbox':bbox})
            return pd.DataFrame(predictions)
        
    def calcualte_iou(self,boxA,boxB):
        xA=max(boxA[0],boxB[0])
        yA=max(boxA[1],boxB[1])
        xB=min(boxA[2],boxB[2])
        yB=min(boxA[3],boxB[3])

        interArea = max(0,xB-xA)*max(0,yB-yA)
        boxAArea = (boxA[2] - boxA[0])* (boxA[3]-boxA[1])
        boxBArea = (boxB[2] - boxB[0])* (boxB[3]-boxB[1])
        return interArea/float(boxAArea +boxBArea - interArea)
    
    def classification_report(self,true_df,predicted_df):
        merged_df= pd.merge(true_df, predicted_df, on='filename',suffixes=('_true','_pred'))
        accuracy = accuracy_score(merged_df['class_true'],merged_df['class_pred'])
        return {'accuracy': accuracy}
    
    def iou_report(self,true_df,predicted_df):
        merged_df = pd.merge(true_df,predicted_df,on='filename',suffixes=('_true','_pred'))
        merged_df['iou']= merged_df.apply(
            lambda row: self.calcualte_iou(row['bbox_true'],row['bbox_pred']),axis=1
        )
        return merged_df['iou'].mean()