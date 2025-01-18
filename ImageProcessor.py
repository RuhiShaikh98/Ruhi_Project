import os
import pandas as pd
import numpy as np
import random
import shutil
import cv2
from PIL import Image, ImageEnhance, ImageOps
from scipy.stats import norm
import matplotlib.pyplot as plt


class ImageProcessor:

    def __init__(self,train_df,val_df,test_df, dataset_dir):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
         
        self.train_images = os.path.join(dataset_dir,'IMAGES','TRAIN')
        self.val_images = os.path.join(dataset_dir,'IMAGES','VAL')
        self.test_images = os.path.join(dataset_dir,'TEST')
        self.processed_dir = os.path.join(dataset_dir,'ProcessedImages')
        self.augmented_dir=os.path.join(dataset_dir,'AUGMENTED')


    def clear_directory(self,path):
        if not os.path.isdir(path):
            return
        for file in os.listdir(path):
            file_path = os.path.join(path,file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")


    def preprocess_spacecraft_images(self,image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

        # Perform Otsu Thresholding
        _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
        closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed_img

    def process_and_copy_images(self,selected_classes,dataset ='train',skip=False):

        if skip:
            return
        
          
        input_dir = getattr(self,f"{dataset}_images")
        df = getattr(self, f"{dataset}_df")
        output_dir =os.path.join(self.processed_dir,dataset,'IMAGES')
        label_output_dir =os.path.join(self.processed_dir,dataset,'LABELS')

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(label_output_dir,exist_ok=True)
        self.clear_directory(output_dir)

        selected_classes_set = set(selected_classes)

        for _, row in df.iterrows():
            if row['class'] in selected_classes_set:
                input_image_path = os.path.join(input_dir, row['filename'])
                processed_image = self.preprocess_spacecraft_images(input_image_path)
                output_image_path = os.path.join(output_dir, row['filename'])
                cv2.imwrite(output_image_path,processed_image)
            
            label_input_path = os.path.join(self.train_images,os.path.splitext(row['filename'])[0]+".txt")
            label_output_path = os.path.join(label_output_dir,os.path.splitext(row['filename'])[0]+".txt")
            shutil.copy(label_input_path, label_output_path)



    def new_backgrounds_augment(self,df,skip =False,bg_dir =None):

        if skip:
            return df

        image_dir= self.train_images
        output_dir=os.path.join(self.augmented_dir,'TRAIN','IMAGES')
        os.makedirs(output_dir,exist_ok=True)
        bg_dir=bg_dir or os.path.join(self.processed_dir,'syntheticBackgrounds')
        new_data = []

        for _, row in df.iterrows():
            # Read the image
            image_path = os.path.join(image_dir, row['filename'])
            image = cv2.imread(image_path)

            # Crop the object
            y1, x1, y2, x2 = map(int, row['bbox'])
            cropped = image[y1:y2, x1:x2]

            # Select a random background
            bg_name = random.choice(os.listdir(bg_dir))
            bg_image = cv2.imread(os.path.join(bg_dir, bg_name))
        
            
            # Resize cropped object
            obj_h, obj_w = cropped.shape[:2]
            bg_h, bg_w = bg_image.shape[:2]
            scale = min(bg_w / obj_w, bg_h / obj_h)
            obj_w,obj_h = int(obj_w*scale), int(obj_h*scale)
            cropped_resized = cv2.resize(cropped, (obj_w,obj_h))

            #Place object randomly on the background
            x_offset=random.randint(0,bg_w-obj_w)
            y_offset=random.randint(0,bg_h-obj_h)
            alpha=0.7
            bg_image[y_offset:y_offset + obj_h, x_offset:x_offset +obj_w] =cv2.addWeighted(
                bg_image[y_offset:y_offset +obj_h, x_offset:x_offset +obj_w],
                1-alpha,
                cropped_resized,
                alpha,
                0
            )

            #Save augmented image
            new_filename=f"augmented_{row['filename']}"
            cv2.imwrite(os.path.join(output_dir,new_filename),bg_image)
       
            # Save YOLO bounding box
            x_center = (x_offset + (obj_w / 2))/bg_w
            y_center = (y_offset + (obj_h / 2))/bg_h
            width = obj_w/bg_w
            height = obj_h/bg_h
            new_data.append({'filename':new_filename,'class':row['class'],'bbox':[x_center,y_center,width,height]})
        
        return pd.DataFrame(new_data)

 
    def crop_flip_augment(self,df, skip=False):
       
        if skip:
            return df
        
        image_dir=self.train_images
        output_dir=os.path.join(self.augmented_dir,'TRAIN','IMAGES')
        os.makedirs(output_dir,exist_ok=True)
        new_data = []

        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row['filename'])
            image = Image.open(image_path)
            cropped = image.crop(row['bbox'])

            if np.random.random() > 0.5:
                cropped_object = ImageOps.mirror(cropped)
            if np.random.random() > 0.5:
                cropped_object = ImageOps.flip(cropped)
            if np.random.random() > 0.5:
                cropped_object = cropped.rotate(random.choice([90, 180, 270]))
                                                
           
            new_filename = f"cropped_{row['filename']}"
            cropped.save(os.path.join(output_dir, new_filename))
            
            new_bbox=[0.5,0.5,cropped.size[0]/1024,cropped.size[1]/1024]
            new_data.append({'filename':new_filename,'class':row['class'],'bbox':new_bbox})
    
        return pd.DataFrame(new_data)






