import os
import pandas as pd
import shutil
import numpy as np

class Sampler:
    
   def __init__(self,train_df,val_df,test_df,train_samples=100,val_samples=20,test_samples=20):

        #Initialize the sampler with DataFrames and Sample sizes.
        self.sampled_train_df = self.sample_dataframe_by_class(train_df, 'class', train_samples)
        self.sampled_val_df = self.sample_dataframe_by_class(val_df, 'class', val_samples)
        self.sampled_test_df = self.sample_dataframe_by_class(test_df, 'class', test_samples)

        pass

   def sample_dataframe_by_class(self,input_df, class_column, samples_per_class):
        #Sample rows from a DataFrame by class with a fixed number of samples per class.
        np.random.seed(42)
        grouped = input_df.groupby(class_column)
        
        # Initialize an empty DataFrame to store the samples
        sampled_df = pd.DataFrame()

        # Sample rows from each group
        for name, group in grouped:
            if len(group) >= samples_per_class:
                sampled_group = group.sample(samples_per_class)
            else:
                #If not enough samples, take all available rows
                sampled_group = group
            sampled_df = pd.concat([sampled_df, sampled_group])

        # Reset the index of the final sampled DataFrame
        sampled_df = sampled_df.reset_index(drop=True)

        return sampled_df

    

   
   def copy_images_and_labels(self,dataframe, source_image_dir, target_image_dir, label_dir, target_label_dir):

        #Copy images and their corresponding labels to targeet directories.

        # Create the target image directory if they doesn't exist
        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)

        # Clear the target directories 
        for target_dir in [target_image_dir, target_label_dir]:
            for filename in os.listdir(target_dir):
                file_path = os.path.join(target_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        #Copy images and labels
        for _, row in dataframe.iterrows():
            #Copy image
            image_filepath = os.path.join(source_image_dir, row['filename'])
            target_image_path = os.path.join(target_image_dir, row['filename'])
            if os.path.exists(image_filepath):
                shutil.copy(image_filepath, target_image_path)

            #Copy label
            label_filename = os.path.splitext(row['filename'])[0] + ".txt"
            label_filepath = os.path.join(label_dir, label_filename)
            target_label_path = os.path.join(target_label_dir,label_filename)
            if os.path.exists(label_filepath):
                shutil.copy(label_filepath, target_label_path)
   
   def make_samples(self, dataset_dir, skip=False):
       #Create training,validation, and testing samples.
       
       if not skip:
            # Training samples
            train_image_dir = os.path.join(dataset_dir,'IMAGES','TRAIN')
            train_label_dir = os.path.join(dataset_dir, 'LABELS','train')
            target_train_image_dir = os.path.join(dataset_dir, 'SAMPLES','TRAIN','IMAGES')
            target_train_label_dir = os.path.join(dataset_dir, 'SAMPLES','TRAIN','LABELS')
            self.copy_images_and_labels(self.sampled_train_df, train_image_dir, target_train_image_dir, train_label_dir, target_train_label_dir)
            print('Training samples created.')
            
            #Validation samples
            val_image_dir = os.path.join(dataset_dir, 'IMAGES', 'VAL')
            val_label_dir = os.path.join(dataset_dir, 'LABELS', 'VAL')
            
            target_val_image_dir = os.path.join(dataset_dir,'SAMPLES','VAL','IMAGES')
            target_val_label_dir = os.path.join(dataset_dir,'SAMPLES','VAL','LABELS')
            self.copy_images_and_labels(self.sampled_val_df, val_image_dir, target_val_image_dir, val_label_dir, target_val_label_dir)
            print('validation samples created.')
            
            #Testing samples
            test_image_dir = os.path.join(dataset_dir,'TEST')
            test_label_dir = os.path.join(dataset_dir,'LABELS','test')
            target_test_image_dir = os.path.join(dataset_dir,'SAMPLES','TEST','IMAGES')
            target_test_label_dir = os.path.join(dataset_dir,'SAMPLES','TEST','LABELS')
            self.copy_images_and_labels(self.sampled_test_df, test_image_dir, target_test_image_dir, test_label_dir, target_test_label_dir)
            print('Testing samples created.')
            
