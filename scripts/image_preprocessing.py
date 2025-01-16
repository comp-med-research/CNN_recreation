import os
import pydicom
import random
import numpy as np
from PIL import Image

def resize_dicom(input_path, train_output_path, test_output_path,  target_size=(224, 224), train_ratio=0.8):
    """
    Resize DICOM images to a uniform size and save the resized image.

    Parameters:
    - input_path (str): Path to the folder containing DICOM files.
    - train_output_path (str): Path to save the resized train images.
    - test_output_path (str): Path to save the resized test images.
    - target_size (tuple): Target size (width, height) for resizing. Default is (224, 224).
    - train_ratio (float): Ratio of images to assign to the train directory (default is 0.8).
    """
    # Ensure the output directories exist
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    # Get list of DICOM files
    dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]

    # Shuffle files for randomness
    random.shuffle(dicom_files)

    # Split into train and test sets
    train_count = int(len(dicom_files) * train_ratio)
    train_files = dicom_files[:train_count]
    test_files = dicom_files[train_count:]

    print(f"Total files: {len(dicom_files)}, Train: {len(train_files)}, Test: {len(test_files)}")

    # Process files
    for filename in dicom_files:
        file_path = os.path.join(input_path, filename)
        output_dir = train_output_path if filename in train_files else test_output_path

        try:
            # Read and process DICOM
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array

            # Normalize the pixel array
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
            pixel_array = pixel_array.astype(np.uint8)

            # Resize image
            image = Image.fromarray(pixel_array)
            image_resized = image.resize(target_size, Image.Resampling.LANCZOS)

            # Save resized image
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_resized.png")
            image_resized.save(output_file_path)

            print(f"Resized and saved to {'train' if output_dir == train_output_path else 'test'}: {output_file_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        break

# Example usage
input_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_sample_cohort_mris/002_S_4262/MPRAGE/2012-10-25_12_02_28.0/I346110"  # Replace with the folder containing DICOM files
train_output_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/train"  # Replace with the folder to save resized images
test_output_directory = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/test"  # Replace with the folder to save resized images
resize_dicom(input_directory, train_output_directory, test_output_directory)
