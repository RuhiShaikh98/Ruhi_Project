import sys
import os
from Preprocessor import Preprocessor
from Sampler import Sampler 
from ObjectDetection import ObjectDetector 
from ImageProcessor import ImageProcessor 
from ultralytics import YOLO
import pandas as pd

def main():


########################################Data Preparation######################################################

    
    #Set the base directory and dataset paths
    curr_dir ='/home/ruhi_musa/'
    dataset_dir = os.path.join(curr_dir,'DATASET')

    train_image_folder = os.path.join(dataset_dir, 'IMAGES','TRAIN')
    val_image_folder = os.path.join(dataset_dir, 'IMAGES','VAL')
    test_image_folder = os.path.join(dataset_dir, 'TEST')
    
    train_data_csv = os.path.join(dataset_dir,'LABELS','train.csv')
    val_data_csv = os.path.join(dataset_dir,'LABELS','val.csv')

    #Initialize Preprocessor with your dataset
    processor = Preprocessor(
        train_image_folder=train_image_folder,
        val_image_folder=val_image_folder,
        train_csv=train_data_csv,
        val_csv=val_data_csv)
    
    #Convert CSV annotations to YOLO format
    processor.convert_csv_to_yolo_format(
        csv_path=train_data_csv,
        output_dir=os.path.join(dataset_dir,'LABELS','train'),
        image_dir=train_image_folder
    )

    processor.convert_csv_to_yolo_format(
        csv_path=val_data_csv,
        output_dir=os.path.join(dataset_dir,'LABELS','val'),
        image_dir=val_image_folder
    )
    

    ########################################Sampling############################################################
    
    sampler = Sampler(
        train_df=processor.train_df,
        val_df=processor.val_df,
        test_df=pd.DataFrame(), #Placeholder for test_df if not processed yet
        train_samples=100,
        val_samples=50,
        test_samples=50
    )
    sampler.make_samples(dataset_dir=dataset_dir)

    ########################################Augmentation########################################################
    #Initialize ImageProcessor
    img_processor = ImageProcessor(
        train_df=processor.train_df,
        val_df=processor.val_df,
        test_df=pd.DataFrame(), #Placeholder for test_df
        dataset_dir=dataset_dir
    )
    
    #Preprocess images
    img_processor.process_and_copy_images(['smart1','cheops','lisa_pathfinder','debris','proba_3_ocs','soho','earth_observation_sat_1','proba_2','xmm_newton','double_star'],dataset='train')

    #Augmentation
    aug_train=img_processor.new_backgrounds_augment(processor.train_df,skip=False)
    aug_crop=img_processor.crop_flip_augment(processor.train_df,skip=False)
    aug_train.to_csv(os.path.join(dataset_dir,'LABELS','train_augmented.csv'),index=False)

    processor.convert_csv_to_yolo_format(
        csv_path=os.path.join(dataset_dir,'LABELS','train_augmented.csv'),
        output_dir=os.path.join(dataset_dir, 'LABELS','augmented_train'),
        image_dir=os.path.join(dataset_dir,'AUGMENTED','TRAIN')
    )

    ########################################Training############################################################

    #Object Detection Training
    detector = ObjectDetector(
        data_config = os.path.join(curr_dir,'config.yaml'), 
        model_path =os.path.join(curr_dir,'models','yolov8_model.pt')
    )

    detector.train_yolo_model(
        epochs=50,
        patience =10,
        imgsz = 640, 
        batch_size=16,
        lr0=0.001 ,
        optimizer = 'SGD') 
                        
if __name__ == "__main__":
    main()