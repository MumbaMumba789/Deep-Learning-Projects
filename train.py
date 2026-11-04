
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from data_setup import create_dataloaders
from model_builder import create_model
from engine import train_step, test_step


# Configuration
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# Data
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir="data/train",
    test_dir="data/test",
    batch_size=BATCH_SIZE
)

print("Classes:", class_names)


# Model
model = create_model(num_classes=len(class_names))
model = model.to(device)


# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# TensorBoard
writer = SummaryWriter(log_dir="runs/books_bags")


# Training loop
for epoch in range(EPOCHS):

    print(f"\nEpoch: {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = train_step(
        model, train_dataloader, loss_fn, optimizer, device
    )

    test_loss, test_acc = test_step(
        model, test_dataloader, loss_fn, device
    )

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")

    writer.add_scalars(
        "Loss",
        {"Train": train_loss, "Test": test_loss},
        epoch
    )

    writer.add_scalars(
        "Accuracy",
        {"Train": train_acc, "Test": test_acc},
        epoch
    )

    writer.flush()


# Save model
torch.save(model.state_dict(), "books_bags_model.pth")

writer.close()

print("\nTraining complete!")

