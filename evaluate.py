import os
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path,test_images_dir,true_labels_path):
    model = YOLO(model_path)
    results= model.predict(source=test_images_dir,save=False)

    #Read ground truth labels
    true_labels = pd.read_csv(true_labels_path)
    y_true = true_labels['class'].tolist()

    #Extract predictions
    y_pred = [result.boxes.cls[0] if hasattr(result.boxes, 'cls') and result.boxes.cls.size >0 else "no detection" for result in results]

    #Generate a classification report
    print("Classification Report:")
    print(classification_report(y_true,y_pred,zero_divisions=1))

    #Generate a confusion matrix
    cm = confusion_matrix(y_true,y_pred,labels=[model.names[int(i)] for i in range(len(model.names))])
    sns.heatmap(cm, annot=True, fmt='d',xticklabels=model.names, yticklabels=model.names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    model_path = "/home/ruhi_musa/models/yolov8_model.pt"
    test_images_dir = "/home/ruhi_musa/DATASET/TEST"
    true_labels_path = "/home/ruhi_musa/DATASET/LABELS/test.csv"
    evaluate_model(model_path,test_images_dir,true_labels_path)
    