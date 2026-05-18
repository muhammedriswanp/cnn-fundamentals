import torch
import torchvision
import mlflow
import copy
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from model import CNN

torch.manual_seed(42)

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label
    

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(28,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# download FashionMNIST
full_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=None
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# split train into train + validation (80/20)
train_size = int(0.8 * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

train_dataset = TransformDataset(train_subset, train_transform)
val_dataset   = TransformDataset(val_subset,   test_transform)


batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 5

kernel_size, stride= 3,2

mlflow.set_experiment("cnn-fashionmnist")

# track best model across experiments
best_val_acc = 0
best_model_state = None

model = CNN(kernel_size=kernel_size, stride=stride)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_name = optimizer.__class__.__name__
run_name = f"kernel{kernel_size}_stride{stride}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "kernel_size":  kernel_size,
        "stride":       stride,
        "lr":           lr,
        "epochs":       epochs,
        "optimizer":    optimizer_name,
        "architecture": str(model)})
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (output.argmax(1) == labels).sum().item()

        train_acc  = correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss_total, val_correct = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                loss   = criterion(output, labels)

                val_loss_total += loss.item()
                val_correct    += (output.argmax(1) == labels).sum().item()

        val_acc  = val_correct / len(val_dataset)
        val_loss = val_loss_total / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc
        }, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("Training complete!")
        
    best_model = CNN(kernel_size=kernel_size, stride=stride)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    correct, total_loss = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = best_model(images)
            loss   = criterion(output, labels)

            total_loss += loss.item()
            correct    += (output.argmax(1) == labels).sum().item()

    test_acc  = 100 * correct / len(test_dataset)
    test_loss = total_loss / len(test_loader)

    mlflow.log_metric("test_acc", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.pytorch.log_model(best_model, f"kernel{kernel_size}_stride{stride}")

    print(f"Test Accuracy: {test_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), f"models/kernel{kernel_size}_stride{stride}.pth")
