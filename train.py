import torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
from model import BReGNeXt
from tqdm import tqdm

def train_model(train_loader, val_loader, model, epochs=10, lr=0.001, accumulation_steps=8, run_name='run'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decay lr every 5 epochs

    scaler = GradScaler()  # For mixed precision training
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Ensure the output directory structure is correct
    output_dir = os.path.join("outputs", run_name)  # Correct path without double "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    # Lists to store loss/accuracy for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

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

            # Log gradients and parameters every 10 batches (you can adjust this)
            if (i + 1) % 10 == 0:
                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Check if gradients exist
                        writer.add_histogram(f"gradients/{name}", param.grad, epoch * len(train_loader) + i)

                # Log parameter distributions
                for name, param in model.named_parameters():
                    writer.add_histogram(f"parameters/{name}", param, epoch * len(train_loader) + i)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Disable mixed precision on specific layers that may cause issues
                with autocast(enabled=False):  # Disable autocast for validation
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Log Learning Rate (from the scheduler)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Append the metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
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

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    writer.close()  # Close the TensorBoard writer

    return model
