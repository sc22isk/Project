import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

train_dir = './vggdata/seg_train/seg_train'
val_dir = './vggdata/seg_test/seg_test'
classes = os.listdir(train_dir)
print("Classes: ", classes)
batch_size = 64
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

total_train = len(train_data_loader)
total_val = len(val_data_loader)

print("Total training data batches: ", total_train)
print("Total validation data batches: ", total_val)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

total_train = len(train_data_loader)
total_val = len(val_data_loader)

print("Total training data batches: ", total_train)
print("Total validation data batches: ", total_val)

def plotImages(img_arr):
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

data_iter = iter(train_data_loader)
training_images, _ = next(data_iter)
plotImages(training_images[:4])



# Load base model as VGG19
base_model = models.vgg19(pretrained=True)
print("Total number of layers in base model: ", len(base_model.features))

# Freeze training for 15 layers and use the rest with trainable weights
for param in base_model.features[:15].parameters():
    param.requires_grad = False

for param in base_model.features[15:].parameters():
    param.requires_grad = True

num_features = base_model.classifier[0].in_features
base_model.classifier = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 100),
    nn.ReLU(inplace=True),
    nn.Linear(100, 100),
    nn.Softmax(dim=1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = base_model.to(device)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()  # Set the model to training mode
    
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / total_train
    epoch_acc = 100 * correct / total
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_epoch_loss = val_loss / total_val
    val_epoch_acc = 100 * val_correct / val_total
    
    print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")
