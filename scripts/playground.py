import pandas as pd 

# read in cohort
# df = pd.read_csv("/Users/halimat/Downloads/ADNI1_Annual_2_Yr_3T_1_16_2025 (1).csv")

# print(len(df))
# print(len(df.Subject.unique()))

# df2 = df.drop_duplicates(subset='Subject', keep='first', inplace=False)

# print(len(df2))

# print(df2.head())

# df2.to_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/nii_sample_cohort.csv", index=False)

# df = pd.read_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/nii_sample_cohort.csv")

# print(df.groupby("Group").size())

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the .nii file
file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_nii_sample_cohort_mris/051_S_1331/MPR__GradWarp__B1_Correction__N3__Scaled/2009-07-13_09_41_30.0/I174851/ADNI_051_S_1331_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20100527161945092_S69898_I174851.nii"
# file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_nii_sample_cohort_mris/136_S_0579/MPR____N3__Scaled/2008-08-18_11_18_37.0/I120618/ADNI_136_S_0579_MR_MPR____N3__Scaled_Br_20081014114857145_S55502_I120618.nii"
# file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/ADNI_nii_sample_cohort_mris/136_S_0196/MPR____N3__Scaled_2/2006-05-01_11_18_56.0/I119725/ADNI_136_S_0196_MR_MPR____N3__Scaled_2_Br_20081008133847111_S13831_I119725.nii"
img = nib.load(file_path)

# Get the data as a numpy array
data = img.get_fdata()

# Print the shape of the data
print(f"Data shape: {data.shape}")  # (e.g., 256, 256, 150 for a 3D image)

# # Display middle slice from each plane
# z_index = data.shape[2] // 2  # Axial slice
# y_index = data.shape[1] // 2  # Coronal slice
# x_index = data.shape[0] // 2  # Sagittal slice

# # Plot the slices
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(data[:, :, z_index], cmap="gray", origin="lower")
# axes[0].set_title("Sagittal Plane")
# axes[1].imshow(data[:, y_index, :], cmap="gray", origin="lower")
# axes[1].set_title("Coronal Plane")
# axes[2].imshow(data[x_index, :, :], cmap="gray", origin="lower")
# axes[2].set_title("Axial Plane")
# plt.show()

# Select a slice (e.g., axial plane)
x_index = data.shape[0] // 2  # Middle slice along the x-axis
axial_slice = data[x_index,:, : ]

# Print the shape of the axial slice
print(f"Data shape: {axial_slice.shape}")  # (e.g., 128, 256, 150 for a 3D image)
print(f"Data shape: {data[x_index, :, :].shape}")  # (e.g., 128, 256, 150 for a 3D image)


# Display the original and rotated slice
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(data[x_index, :, :], cmap="gray", origin="lower")
axes[0].set_title("Original Slice")
axes[1].imshow(np.rot90(data[x_index, :, :],2), cmap="gray", origin="lower")
axes[1].set_title("Rotated Slice (180°)")
plt.show()
