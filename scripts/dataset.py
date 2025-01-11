from torch.utils.data import Dataset

class DICOMDataset(Dataset):
    def __init__(self, dicom_dir, target_size=(224, 224)):
        self.dicom_dir = dicom_dir
        self.files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        dicom_path = os.path.join(self.dicom_dir, self.files[idx])
        tensor = process_dicom(dicom_path, self.target_size)
        return tensor

# Example usage
dataset = DICOMDataset(dicom_directory)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch.shape)  # Shape: (32, 1, 224, 224)
