import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset to load images and labels.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def prepare_data(image_folder, val_split=0.2, batch_size=32):
    # List all images in the training directory
    image_paths = []
    labels = []
    label_map = {}

    current_label = 0
    train_folder = os.path.join(image_folder, "train")
    test_folder = os.path.join(image_folder, "test")

    # Traverse the train folder and collect images and labels
    for class_name in os.listdir(train_folder):
        class_path = os.path.join(train_folder, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                if image_name.endswith(('jpg', 'png', 'jpeg')):
                    image_paths.append(os.path.join(class_path, image_name))
                    labels.append(current_label)
            label_map[class_name] = current_label
            current_label += 1

    # Split the data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=42
    )

    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])
    
    # Create datasets for training and validation
    train_dataset = CustomImageDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomImageDataset(val_paths, val_labels, transform=transform)

    # Create DataLoader objects for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the test set (without a split)
    test_image_paths = []
    test_labels = []
    for class_name in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                if image_name.endswith(('jpg', 'png', 'jpeg')):
                    test_image_paths.append(os.path.join(class_path, image_name))
                    test_labels.append(label_map[class_name])

    test_dataset = CustomImageDataset(test_image_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_map
