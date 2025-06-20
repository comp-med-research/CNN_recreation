import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import lightning as L 
# from lightning.pytorch import Trainer
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint
# from model import CNN1, CNN2, TwoTowerCNN
# from dataset import AlzheimerDataset

# # Import your TwoTowerCNN, CNN1, CNN2, and DICOMDataset
# # from your_model_file import TwoTowerCNN, CNN1, CNN2
# # from your_dataset_file import DICOMDataset, process_dicom

# # Define TwoTowerCNN Lightning Module
# class TwoTowerModel(L.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.cnn1 = CNN1()
#         self.cnn2 = CNN2()
#         self.model = TwoTowerCNN(self.cnn1, self.cnn2)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         images, labels = batch
#         images, labels = images.float(), labels.long()
#         outputs = self(images)
#         loss = self.criterion(outputs, labels)

#         # Calculate accuracy
#         _, preds = torch.max(outputs, 1)
#         acc = (preds == labels).float().mean()

#         # Calculate precision
#         # _, preds = torch.max(outputs, 1)
#         # precision = (preds == labels).float().mean()

#         # # Calculate recall
#         # _, preds = torch.max(outputs, 1)
#         # recall = (preds == labels).float().mean()

#         # # Calculate F1
#         # _, preds = torch.max(outputs, 1)
#         # F1 = (preds == labels).float().mean()

#         # Log training metrics
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("train_precision", precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("train_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("train_F1", F1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, labels = batch
#         images, labels = images.float(), labels.long()
#         outputs = self(images)
#         loss = self.criterion(outputs, labels)

#         # Calculate accuracy
#         _, preds = torch.max(outputs, 1)
#         acc = (preds == labels).float().mean()

#         # Calculate precision
#         # _, preds = torch.max(outputs, 1)
#         # precision = (preds == labels).float().mean()

#         # # Calculate recall
#         # _, preds = torch.max(outputs, 1)
#         # recall = (preds == labels).float().mean()

#         # # Calculate F1
#         # _, preds = torch.max(outputs, 1)
#         # F1 = (preds == labels).float().mean()

#         # Log validation metrics
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("val_precision", precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("val_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("val_F1", F1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
#         return optimizer

# # Configuration
# config = {
#     "learning_rate": 0.001,
#     "batch_size": 32,
#     "num_epochs": 10,
#     "weight_decay": 1e-5,
#     "target_size": (224, 224),
#     "num_classes": 5
# }

# # Transformations
# transform = transforms.Compose([
#     transforms.Resize(config["target_size"]),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),
# ])

# def main():
#     # Setting all the random seeds to the same value.
#     # This is important in a distributed training setting.
#     # Each rank will get its own set of initial weights.
#     # If they don't match up, the gradients will not match either,
#     # leading to training that may not converge.
#     L.seed_everything(1234)

#     # Wandb Logger
#     wandb_logger = WandbLogger(project="twotower-cnn", entity="halimatafolabi-attenxion")
#     wandb_logger.log_hyperparams(config)

#     # Paths
#     train_dir = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/train"  # Replace with your train directory path
#     # test_dir = "/path/to/processed/test"   # Replace with your test directory path

#     # Define transformations
#     data_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
#     ])

#     # Create training dataset and split into train and validation subsets
#     full_train_dataset = AlzheimerDataset(train_dir, transform=data_transforms)
#     train_size = int(0.8 * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

#     # Create test dataset
#     # test_dataset = AlzheimerDataset(test_dir, transform=data_transforms)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
#     # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#     # Print dataset sizes
#     print(f"Training set size: {len(train_dataset)}")
#     print(f"Validation set size: {len(val_dataset)}")
#     # print(f"Test set size: {len(test_dataset)}")

#     # Initialize the Lightning Model
#     lightning_model = TwoTowerModel(config)

#     # Checkpoints
#     callbacks = [
#             ModelCheckpoint(
#                 dirpath="checkpoints",
#                 every_n_train_steps=100,
#                 monitor="val_loss", mode="min"
#             ),
#         ]

#     # Define the PyTorch Lightning Trainer
#     trainer = Trainer(
#         max_epochs=config["num_epochs"],
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         logger=wandb_logger,
#         log_every_n_steps=10,
#         deterministic=True,
#         strategy="ddp", 
#         callbacks=callbacks
#     )

#     # Train the Model
#     trainer.fit(lightning_model, train_loader, val_loader)

#     # Finish W&B Logging
#     wandb_logger.experiment.finish()


# if __name__ == "__main__":
#     main()
