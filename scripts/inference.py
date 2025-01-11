import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from model import TwoTowerCNN, CNN1, CNN2  # Replace with your actual import paths
from dataset import DICOMDataset  # Replace with your actual import paths

# Define LightningModule for Evaluation
class TwoTowerEvaluator(pl.LightningModule):
    def __init__(self, config, model_checkpoint_path, output_dir="output"):
        super().__init__()
        self.config = config

        # Initialize model
        self.cnn1 = CNN1()
        self.cnn2 = CNN2()
        self.model = TwoTowerCNN(self.cnn1, self.cnn2)

        # Load model checkpoint
        self.model.load_state_dict(torch.load(model_checkpoint_path))
        self.model.eval()

        # Metrics storage
        self.preds = []
        self.labels = []
        self.output_dir = output_dir

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.float(), labels.long()

        # Forward pass
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels
        self.preds.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())

    def test_epoch_end(self, outputs):
        # Convert stored predictions and labels to numpy arrays
        predictions = np.array(self.preds)
        ground_truth = np.array(self.labels)

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average="weighted", zero_division=0)
        recall = recall_score(ground_truth, predictions, average="weighted", zero_division=0)
        f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        balanced_acc = balanced_accuracy_score(ground_truth, predictions)
        mcc = matthews_corrcoef(ground_truth, predictions)

        # Log metrics
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_balanced_accuracy", balanced_acc)
        self.log("test_mcc", mcc)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        # Generate classification report
        class_report = classification_report(ground_truth, predictions, zero_division=0)

        # Save classification report to a file
        df = pd.DataFrame(class_report).transpose()
        report_path = f"{self.output_dir}/classification_report.csv" #probs need to add checkpoint of trained model and date + time or something to keep it unique
        df.to_csv(report_path)

        print(f"\nClassification report saved to {report_path}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(ground_truth, predictions)
        conf_matrix_path = f"{self.output_dir}/confusion_matrix.png"

        # Save confusion matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(self.config["num_classes"]), yticklabels=range(self.config["num_classes"]))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(conf_matrix_path)
        plt.close()

        print(f"Confusion matrix saved to {conf_matrix_path}")

        # AUC 
        

        # log images to w&b to visualize misclassified images

        # view grad-cam of images, especially misclassified images, save these or see if they can be logged on weights and biases

# Configuration
config = {
    "batch_size": 32,
    "target_size": (224, 224),
    "num_classes": 5,
    "model_checkpoint_path": "path/to/your/checkpoint.pth",  # Replace with your checkpoint path
    "output_dir": "output"  # Directory to save the classification report
}

# Transformations
transform = transforms.Compose([
    transforms.Resize(config["target_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Test dataset and dataloader
test_dataset = DICOMDataset(dicom_dir="path/to/test", target_size=config["target_size"])  # Replace with your test dataset path
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize the evaluator
evaluator = TwoTowerEvaluator(config, config["model_checkpoint_path"], config["output_dir"])

# Define PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    deterministic=True,
    log_every_n_steps=10,
)

# Run testing
trainer.test(evaluator, test_loader)
