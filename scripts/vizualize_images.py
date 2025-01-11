import pydicom
import matplotlib.pyplot as plt

def visualize_dicom(file_path):
    """
    Visualize a .dcm (DICOM) file.

    Parameters:
    file_path (str): Path to the DICOM file.
    """
    try:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(file_path)

        # Extract the pixel array
        pixel_array = dicom_data.pixel_array
        print(pixel_array.shape)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(pixel_array, cmap='gray')
        # plt.title(f"DICOM Image: {file_path}")
        # plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error visualizing DICOM file: {e}")

# Example usage
if __name__ == "__main__":
    dicom_file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_sample_cohort_mris/002_S_4262/MPRAGE/2012-10-25_12_02_28.0/I346110/ADNI_002_S_4262_MR_MPRAGE_br_raw_20121112110714262_110_S174193_I346110.dcm"
    # dicom_file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/test2.dcm"  # Replace with the path to your .dcm file
    visualize_dicom(dicom_file_path)

# # slide through dicom images using a slide bar 
# plt.figure(1)
# def dicom_animation(x):
#     plt.imshow(patient_pixels[x], cmap = plt.cm.gray)
#     return x
# interact(dicom_animation, x=(0, len(patient_pixels)-1))