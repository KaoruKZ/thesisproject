import torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from model import BReGNeXt  # Assuming your model is saved in 'model.py'
from preprocessing import prepare_data

def train_model(train_loader, val_loader, model, epochs=10, lr=0.001, batch_size=8, accumulation_steps=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decay lr every 5 epochs

    scaler = GradScaler()  # For mixed precision training
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():  # Enable mixed precision for most layers
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Scale the loss for gradient accumulation

            # Accumulate gradients
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:  # Update weights every 'accumulation_steps' batches
                scaler.step(optimizer)
                scaler.update()  # Update the scaler
                optimizer.zero_grad()  # Reset gradients

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
                inputs, labels = inputs.to(device), labels.to(device)

                # Disable mixed precision on specific layers that may cause issues
                with autocast(enabled=False):  # Disable autocast for validation
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
            epochs_without_improvement = 0  # Reset patience counter
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= 3:
            print("Early stopping triggered.")
            break

        scheduler.step()  # Step the learning rate scheduler

        # Clear cache to avoid fragmentation
        torch.cuda.empty_cache()

    print("Training complete.")
    return model


# Main execution
if __name__ == "__main__":
    # Define device (cuda if available, else cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data (adjust batch size)
    image_folder = './fer2013'
    train_loader, val_loader, test_loader, label_map = prepare_data(image_folder, val_split=0.2, batch_size=8)

    # Initialize the model and move it to the device (GPU or CPU)
    model = BReGNeXt(n_classes=len(label_map)).to(device)

    # Train the model
    model = train_model(train_loader, val_loader, model, epochs=10, lr=0.001, batch_size=8)

    # Test evaluation
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Debugging shapes
            print(f"Test - Input shape: {inputs.shape}, Output shape: {outputs.shape}")

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_acc = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_acc:.2f}%")
