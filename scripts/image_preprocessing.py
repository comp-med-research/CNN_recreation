# import os
# import pydicom
# import random
# import numpy as np
# from PIL import Image

# def resize_images(input_path, train_output_path, test_output_path,  target_size=(224, 224), train_ratio=0.8):
#     """
#     Resize DICOM images to a uniform size and save the resized image.

#     Parameters:
#     - input_path (str): Path to the folder containing DICOM files.
#     - train_output_path (str): Path to save the resized train images.
#     - test_output_path (str): Path to save the resized test images.
#     - target_size (tuple): Target size (width, height) for resizing. Default is (224, 224).
#     - train_ratio (float): Ratio of images to assign to the train directory (default is 0.8).
#     """
#     # Ensure the output directories exist
#     os.makedirs(train_output_path, exist_ok=True)
#     os.makedirs(test_output_path, exist_ok=True)

#     # Get list of DICOM files
#     dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]

#     # Shuffle files for randomness
#     random.shuffle(dicom_files)

#     # Split into train and test sets
#     train_count = int(len(dicom_files) * train_ratio)
#     train_files = dicom_files[:train_count]
#     test_files = dicom_files[train_count:]

#     print(f"Total files: {len(dicom_files)}, Train: {len(train_files)}, Test: {len(test_files)}")

#     # Process files
#     for filename in dicom_files:
#         file_path = os.path.join(input_path, filename)
#         output_dir = train_output_path if filename in train_files else test_output_path

#         try:
#             # Read and process DICOM
#             dicom_data = pydicom.dcmread(file_path)
#             pixel_array = dicom_data.pixel_array

#             # Normalize the pixel array
#             pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
#             pixel_array = pixel_array.astype(np.uint8)

#             # Resize image
#             image = Image.fromarray(pixel_array)
#             image_resized = image.resize(target_size, Image.Resampling.LANCZOS)

#             # Save resized image
#             output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_resized.png")
#             image_resized.save(output_file_path)

#             print(f"Resized and saved to {'train' if output_dir == train_output_path else 'test'}: {output_file_path}")
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")
#         break

# # Example usage
# input_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_sample_cohort_mris/002_S_4262/MPRAGE/2012-10-25_12_02_28.0/I346110"  # Replace with the folder containing DICOM files
# train_output_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/train"  # Replace with the folder to save resized images
# test_output_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/test"  # Replace with the folder to save resized images
# resize_images(input_directory, train_output_directory, test_output_directory)

import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt 

# Paths
csv_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/nii_sample_cohort_with_labels.csv"
raw_dir = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_nii_sample_cohort_mris"  # Replace with your actual path
processed_dir = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed"  # Replace with your actual path

# Read the CSV file
df = pd.read_csv(csv_path)

# Iterate through each subject in the raw directory
for subject_dir in os.listdir(raw_dir):
    subject_path = os.path.join(raw_dir, subject_dir)
    
    # Ensure it's a directory
    if os.path.isdir(subject_path):
        # Find corresponding row in the CSV file
        subject_data = df[df["Subject"] == subject_dir]
        
        if not subject_data.empty:
            # Extract label and group information
            label = subject_data["Label"].values[0]  # "train" or "test"
            group = subject_data["Group"].values[0]  # "AD", "MCI", "CN"
            
            
            # Process .nii file
            for root, dirs, files in os.walk(subject_path):
                for file in files:
                    if file.endswith(".nii") or file.endswith(".nii.gz"):
                        nii_file_path = os.path.join(root, file)

                        # Load the .nii file
                        img = nib.load(nii_file_path)
                        data = img.get_fdata()
                        
                        # Example processing: take the middle axial slice
                        x_index = data.shape[0] // 2  # Middle slice along the x-axis
                        axial_slice = data[x_index,:, : ]
                        
                        # Normalize the image
                        axial_slice = (axial_slice - np.min(axial_slice)) / (np.max(axial_slice) - np.min(axial_slice)) * 255.0
                        axial_slice = axial_slice.astype(np.uint8)
                        
                        # Convert to a PIL image
                        image = Image.fromarray(axial_slice)

                        # Resize image
                        image_resized = image.resize((224,224), Image.Resampling.LANCZOS)

                        # Convert image mode to "L" (8-bit grayscale) before saving
                        image_resized = image_resized.convert("L")

                        # Save to the processed directory
                        output_dir = os.path.join(processed_dir, label, group)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_file = os.path.join(output_dir, f"{subject_dir}_processed.png")
                        image_resized.save(output_file)
                        
                        print(f"Processed and saved: {output_file}")
       
                     
