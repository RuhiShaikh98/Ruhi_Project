# MyImageProcessor.py

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image


class Preprocessor:

    def __init__(self,train_image_folder=None,val_image_folder=None,train_csv=None,val_csv=None):
       
        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder
        self.train_csv = train_csv
        self.val_csv = val_csv
        
        #Load and process CSV data if all parameters are provided
        if train_image_folder and val_image_folder and train_csv and val_csv:
            self.train_df, self.val_df = self.load_and_process_csv_data()
       

       
    
    def load_and_process_csv_data(self):
        
        #Load and train val CSVs
        train_data_pd = pd.read_csv(self.train_csv)
        val_data_pd = pd.read_csv( self.val_csv )
        
        #Convert bbox strings to arrays
        train_data_pd['bbox']= train_data_pd['bbox'].apply(self.process_bbox_string)
        val_data_pd['bbox']= val_data_pd['bbox'].apply(self.process_bbox_string)
        
        #Add file paths for images
        train_data_pd['filepath']=train_data_pd['filename'].apply(
            lambda x: os.path.join(self.train_image_folder, x)
        )
        val_data_pd['filepath']=val_data_pd['filename'].apply(
            lambda x: os.path.join(self.val_image_folder, x)
        )

        return train_data_pd,val_data_pd


    def process_bbox_string(self, bbox_str):
        #Convert bounding box string to list of integers
        bbox_coords = [int(coord) for coord in bbox_str.strip('[]').split(',')]
        return np.array(bbox_coords)

    
    def convert_csv_to_yolo_format(self,csv_path,output_dir,image_dir):
        #Load CSV file
        data_df = pd.read_csv(csv_path)
        os.makedirs(output_dir,exist_ok=True)

        for _, row in data_df.iterrows():
            #Extract bounding box and image dimensions
            bbox = self.process_bbox_string(row['bbox'])
            x1, y1, x2, y2 = bbox

            #Calculate YOLO format values
            img=Image.open(os.path.join(image_dir,row['filename']))
            img_width, img_height = img.size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            x_center = x1 + bbox_width / 2
            y_center = y1 + bbox_height / 2

            # Normalize the values to [0,1]
            x_center /= img_width
            y_center /= img_height
            bbox_width /= img_width
            bbox_height /= img_height
                    
            #Write YOLO formatted labels to file
            label_file = os.path.join(output_dir, f"{os.path.splitext(row['filename'])[0]}.txt")
            with open(label_file, "w") as f:
                f.write(f"{row['class']} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    def move_files_to_val_directory(self, val_df, source_directory, destination_directory):
        os.makedirs(destination_directory, exist_ok=True)
        for _, row in val_df.iterrows():
            source_path = os.path.join(source_directory, row['filename'])
            destination_path = os.path.join(destination_directory, row['filename'])
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f"File not found: {source_path}")

                