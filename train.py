 import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from time import time
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # train and test path
    train_root = './Data-Set/train'
    test_root = './Data-Set/test'

    # transformations for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load
    train_dataset = ImageFolder(train_root, transform=transform)
    test_dataset = ImageFolder(test_root, transform=transform)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # model init
    model = CNN(num_classes=len(train_dataset.classes)).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    # training and evaluation loop
    num_epochs = 2
    for epoch in range(num_epochs):
        start_time = time()
        model.train()

        # Training
        train_correct = 0
        train_loss = 0.0
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch"):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            _,preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == y_train).item()
            train_loss += loss.item() * x_train.size(0)

        # training acc and loss
        train_accuracy = train_correct / len(train_dataset)
        train_loss = train_loss / len(train_dataset)

        # evaluation
        model.eval()
        test_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Testing", unit="batch"):
                x_test, y_test = x_test.to(device), y_test.to(device)
                outputs = model(x_test)
                loss = loss_fn(outputs, y_test)

                _,preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == y_test).item()
                test_loss += loss.item() * x_test.size(0)

        # test accuracy and loss
        test_accuracy = test_correct / len(test_dataset)
        test_loss = test_loss / len(test_dataset)

        end_time = time()
        duration = (end_time - start_time) / 60

        print(f"Epoch: {epoch + 1}, Time: {duration:.2f} minutes, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # save
    model_save_path = './model_cnn.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


train_and_evaluate()
