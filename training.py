import torch
import torch.optim as optim
from torch import nn
import time
from model import BReGNeXt  # Assuming your model is saved in 'model.py'
from tqdm import tqdm
from preprocessing import prepare_data


def train_model(train_loader, val_loader, model, epochs=10, lr=0.001):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

    print("Training complete.")
    return model


# Main execution
if __name__ == "__main__":
    image_folder = './fer2013'
    train_loader, val_loader, test_loader, label_map = prepare_data(image_folder, val_split=0.2, batch_size=32)

    model = BReGNeXt(n_classes=len(label_map)).cuda()  # Adjust the number of classes based on your dataset

    model = train_model(train_loader, val_loader, model, epochs=10, lr=0.001)

    # Test evaluation
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_acc = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_acc:.2f}%")
