import os
import pydicom
import numpy as np
from PIL import Image

def resize_dicom(input_path, output_path, target_size=(224, 224)):
    """
    Resize DICOM images to a uniform size and save the resized image.

    Parameters:
    - input_path (str): Path to the folder containing DICOM files.
    - output_path (str): Path to save the resized images.
    - target_size (tuple): Target size (width, height) for resizing. Default is (224, 224).
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.dcm'):  # Check for DICOM files
            file_path = os.path.join(input_path, filename)
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(file_path)
                pixel_array = dicom_data.pixel_array

                # Normalize the pixel array (optional, based on your data needs)
                pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
                pixel_array = pixel_array.astype(np.uint8)

                # Convert to PIL Image for resizing
                image = Image.fromarray(pixel_array)
                image_resized = image.resize(target_size, Image.ANTIALIAS)

                # Save the resized image
                output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_resized.png")
                image_resized.save(output_file_path)

                print(f"Resized and saved: {output_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = "path/to/your/dicom/folder"  # Replace with the folder containing DICOM files
train_output_directory = "path/to/save/resized/images"  # Replace with the folder to save resized images
test_output_directory = "path/to/save/resized/images"  # Replace with the folder to save resized images
resize_dicom(input_directory, train_output_directory, test_output_directory)
