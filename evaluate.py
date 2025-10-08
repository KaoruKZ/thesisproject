# evaluate.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix
from scipy import stats
from data_preprocess import prepare_data  # Assuming this function is in data_preprocess.py
from model import BReGNeXt  # Assuming your model is defined in 'model.py'
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Inference function to evaluate the model on the test dataset
def evaluate_model(test_loader, model, device='cuda'):
    model.to(device)  # Move model to the specified device
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():  # No need to track gradients during inference
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Get class probabilities
            _, predicted = torch.max(outputs, 1)  # Get predicted class labels

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays for further processing
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_preds, all_labels, all_probs

# Confusion matrix plot function
def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Kolmogorov-Smirnov (KS) statistic function
def compute_ks_statistic(true_labels, predicted_probs, num_classes=7):
    ks_values = []
    for i in range(num_classes):
        true_class = (true_labels == i).astype(int)  # Create binary vector for true class
        pred_class_probs = predicted_probs[:, i]  # Get predicted probabilities for the i-th class
        ks_stat, _ = stats.ks_2samp(true_class, pred_class_probs)  # KS test
        ks_values.append(ks_stat)
    return ks_values

# Main execution
if __name__ == "__main__":
    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare test data loader
    image_folder = './fer2013'
    _, _, test_loader, class_names = prepare_data(image_folder, val_split=0.2, batch_size=8)  # Adjust as needed

    # Initialize the model (make sure to set the correct number of output classes)
    model = BReGNeXt(n_classes=7).to(device)

    # Load the trained model weights
    checkpoint_path = './outputs/experiment_1/best_model.pth'  # Path to the saved best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model loaded from {checkpoint_path}")

    # Evaluate the model on the test dataset
    all_preds, all_labels, all_probs = evaluate_model(test_loader, model, device)

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plot and save the Confusion Matrix
    plot_confusion_matrix(conf_matrix, class_names, './outputs/experiment_1')

    # Compute Kolmogorov-Smirnov (KS) statistic
    ks_values = compute_ks_statistic(all_labels, all_probs)
    print(f"Kolmogorov-Smirnov (KS) values for each class: {ks_values}")

    # Print accuracy
    accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")
