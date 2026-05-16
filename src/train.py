import torch
import torchvision
import mlflow
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from model import CNN

# download FashionMNIST
full_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# split train into train + validation (80/20)
train_size = int(0.8 * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

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

experiments = [
    {"kernel_size": 3, "stride": 1},
    {"kernel_size": 5, "stride": 1},
    {"kernel_size": 3, "stride": 2}
]

mlflow.set_experiment("cnn-fashionmnist")

for config in experiments:
    model = CNN(kernel_size=config["kernel_size"], stride=config["stride"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_name = optimizer.__class__.__name__
    run_name = f"kernel{config['kernel_size']}_stride{config['stride']}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "kernel_size":  config["kernel_size"],
            "stride":       config["stride"],
            "lr":           lr,
            "epochs":       epochs,
            "batch_size":   batch_size,
            "train_size":   train_size,
            "val_size":     val_size,
            "optimizer":    optimizer_name,
            "architecture": str(model)
        })

        # training loop
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

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc
            }, step=epoch)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        print("Training complete!")

        # final test evaluation — only run once at the end
        model.eval()
        correct, total_loss = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                output = model(images)
                loss   = criterion(output, labels)
                total_loss += loss.item()
                correct    += (output.argmax(1) == labels).sum().item()

        test_acc  = 100 * correct / len(test_dataset)
        test_loss = total_loss / len(test_loader)
        mlflow.log_metric("test_acc",  test_acc,  step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        print(f"Test Accuracy: {test_acc:.2f}%")

        mlflow.pytorch.log_model(model, f"model_kernel{config['kernel_size']}_stride{config['stride']}")