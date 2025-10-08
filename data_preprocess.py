import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ImageEmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Open the image as RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data(image_folder, val_split=0.2, batch_size=8):
    # Emotion classes
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Data preprocessing: list all image paths and corresponding labels
    image_paths = []
    labels = []

    # Loop over class directories
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_folder, 'train', class_name)  # path to train/{class}
        for filename in os.listdir(class_dir):
            if filename.endswith(".jpg"):  # Only consider .jpg files
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(idx)  # Label is the index of the class

    # Split the data into training and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels)

    # Define transforms for the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create datasets
    train_dataset = ImageEmotionDataset(train_paths, train_labels, transform=transform)
    val_dataset = ImageEmotionDataset(val_paths, val_labels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Emotion map (for reference)
    emotion_map = {idx: class_name for idx, class_name in enumerate(classes)}

    return train_loader, val_loader, emotion_map


def generate_sample_images(dataset, output_dir, num_images_per_class=3):
    images_by_class = {i: [] for i in range(len(dataset.classes))}

    for image, label in dataset:
        if len(images_by_class[label]) < num_images_per_class:
            images_by_class[label].append((image, dataset.classes[label]))

    fig, axes = plt.subplots(len(images_by_class), num_images_per_class, figsize=(12, 8))

    for i, (class_id, images) in enumerate(images_by_class.items()):
        for j, (image, class_name) in enumerate(images):
            ax = axes[i, j] if len(images_by_class) > 1 else axes[j]
            image = image.permute(1, 2, 0).numpy()  # Convert tensor to HWC
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(class_name)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    plt.savefig(os.path.join(output_dir, "sample_images.png"))
    plt.close()
