import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the directory of your training data
train_dir = './dt1/train'
output_dir = './dt1/Train_split'  # Directory to store the split data

# Get a list of class subdirectories
class_subdirectories = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, subdir))]

# Create directories for train, validation, and test data
train_dir_new = os.path.join(output_dir, 'train')
validation_dir_new = os.path.join(output_dir, 'validation')
test_dir_new = os.path.join(output_dir, 'test')

os.makedirs(train_dir_new, exist_ok=True)
os.makedirs(validation_dir_new, exist_ok=True)
os.makedirs(test_dir_new, exist_ok=True)

# Iterate over each class subdirectory
for class_subdir in class_subdirectories:
    class_path = os.path.join(train_dir, class_subdir)
    
    # Get a list of image filenames in the current class
    image_filenames = os.listdir(class_path)
    
    # Split the images into train, validation, and test sets
    train_filenames, temp_filenames = train_test_split(image_filenames, test_size=0.5, random_state=42)
    validation_filenames, test_filenames = train_test_split(temp_filenames, test_size=0.4, random_state=42)
    
    # Move files to their respective directories
    for filename in train_filenames:
        src_path = os.path.join(class_path, filename)
        dst_path = os.path.join(train_dir_new, class_subdir, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    
    for filename in validation_filenames:
        src_path = os.path.join(class_path, filename)
        dst_path = os.path.join(validation_dir_new, class_subdir, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    
    for filename in test_filenames:
        src_path = os.path.join(class_path, filename)
        dst_path = os.path.join(test_dir_new, class_subdir, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

print("Classes: ", classes)
IMG_HEIGHT = 150
IMG_WIDTH = 150

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

train_data = datasets.ImageFolder(train_dir_new, transform=train_transforms)
val_data = datasets.ImageFolder(validation_dir_new, transform=val_transforms)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

epochs = 30
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
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'accuracy': epoch_acc,
    'val_loss': val_epoch_loss,
    'val_accuracy': val_epoch_acc
}, 'saved_model.pth')

# Initialize lists to store true labels and predicted labels
train_true_labels = []
train_pred_labels = []
val_true_labels = []
val_pred_labels = []

# Set the model to evaluation mode
model.eval()

# Collect predictions and true labels for train set
with torch.no_grad():
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        train_true_labels.extend(labels.cpu().tolist())  # Use true labels
        train_pred_labels.extend(predicted.cpu().tolist())  # Use predicted labels

# Collect predictions and true labels for validation set
with torch.no_grad():
    for images, labels in val_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_true_labels.extend(labels.cpu().tolist())  # Use true labels
        val_pred_labels.extend(predicted.cpu().tolist())  # Use predicted labels

# Get class names from your dataset
class_names = os.listdir(train_dir_new)

# Calculate confusion matrices
train_confusion = confusion_matrix(train_true_labels, train_pred_labels)
val_confusion = confusion_matrix(val_true_labels, val_pred_labels)

# Plot confusion matrix for train set
plt.figure(figsize=(10, 8))
sns.heatmap(train_confusion, annot=True, fmt="d", cmap="Blues")
plt.title("Train Set Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.show()

# Plot confusion matrix for validation set
plt.figure(figsize=(10, 8))
sns.heatmap(val_confusion, annot=True, fmt="d", cmap="Oranges")
plt.title("Validation Set Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.show()

