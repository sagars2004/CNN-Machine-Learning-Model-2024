# Sagar Sahu
# This is a CNN machine learning model to classify various elastic stress contour
# images based on their severity (0, 1, 2) for different microstructures.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# Function to read data
def read_files(max_file_num):
    file_num = list(range(max_file_num))
    data_array = np.zeros([1, 150, 150, 4])
    labels_array = np.zeros([1, 1])
    metadata_array = np.zeros([1, 4])
    base_path = '/kaggle/input/dataset-compressed/Sagar'
    for f_num in file_num:
        data_filename = os.path.join(base_path, f"data_file_{f_num+1}.h5")
        label_filename = os.path.join(base_path, f"label_file_{f_num+1}.h5")
        metadata_filename = os.path.join(base_path, f"meta_data_file_{f_num+1}.h5")

        with h5py.File(data_filename, "r") as f:
            a_group_key = list(f.keys())[0]
            loc_data_array = f[a_group_key][()]

        with h5py.File(label_filename, "r") as f:
            a_group_key = list(f.keys())[0]
            loc_labels_array = f[a_group_key][()].T

        with h5py.File(metadata_filename, "r") as f:
            a_group_key = list(f.keys())[0]
            loc_metadata_array = f[a_group_key][()].T

        data_array = np.concatenate((data_array, loc_data_array), axis=0)
        labels_array = np.concatenate((labels_array, loc_labels_array), axis=0)
        metadata_array = np.concatenate((metadata_array, loc_metadata_array), axis=0)

    data_array = np.delete(data_array, 0, axis=0)
    labels_array = np.delete(labels_array, 0, axis=0)
    metadata_array = np.delete(metadata_array, 0, axis=0)

    return data_array, labels_array, metadata_array

# Function to modify labels
def labelHelper(labels_array):
    minValue = np.min(labels_array)
    maxValue = np.max(labels_array)
    labelRange = maxValue - minValue
    modifiedLabels = np.zeros_like(labels_array)
    for i in range(len(labels_array)):
        if labels_array[i, 0] < minValue + labelRange / 3:
            modifiedLabels[i] = 0
        elif labels_array[i, 0] < maxValue - labelRange / 3:
            modifiedLabels[i] = 1
        else:
            modifiedLabels[i] = 2
    return modifiedLabels

# Function to normalize data
def normalize_data(data):
    data_normalized = np.zeros_like(data)
    for channel in range(data.shape[-1]):
        channel_data = data[:, :, :, channel]
        min_val = channel_data.min()
        max_val = channel_data.max()
        data_normalized[:, :, :, channel] = (channel_data - min_val) / (max_val - min_val)
    return data_normalized


# Model layers and features definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 37 * 37, 128)
        #self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 84)
        #self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 3)
        
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reading and preprocessing data
max_file_num = 36
data_set, labels, metadata = read_files(max_file_num)
data_set = normalize_data(data_set)  # Normalize the data
newLabels = labelHelper(labels)

# Converting to PyTorch tensors
x_data = torch.tensor(data_set, dtype=torch.float32).permute(0, 3, 1, 2)  # Change the dimensions to (N, C, H, W)
y_data = torch.tensor(newLabels, dtype=torch.long).squeeze()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variables to store the results
fold_results = {
    'train_loss': [],
    'test_loss': [],
    'train_accuracy': [],
    'test_accuracy': []
}

batch_size = 32
learning_rate = 0.0001

# Printing the input contour images
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Function to read all .h5 files in a directory and extract the data
def read_all_files(directory, max_files=None):
    data_list = []
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            with h5py.File(file_path, "r") as f:
                a_group_key = list(f.keys())[0]
                data = f[a_group_key][()]
                if data.ndim == 4:  # Ensure data is 4-dimensional
                    data_list.append(data)
                #else:
                    #print(f"Skipping file {filename}: data has {data.ndim} dimensions")
            file_count += 1
            if max_files is not None and file_count >= max_files:
                break
    if data_list:
        return np.concatenate(data_list, axis=0)
    else:
        raise ValueError("No valid data found in the specified directory")

# Function to display sample images
def display_sample_images(data_array, num_samples=None):
    if num_samples is None:
        num_samples = data_array.shape[0]  # Display all images if num_samples is not specified
    
    num_cols = 5
    num_rows = (num_samples + num_cols - 1) // num_cols

    plt.figure(figsize=(15, num_rows * 3))
    for i in range(num_samples):
        plt.subplot(num_rows, num_cols, i + 1)
        
        img = data_array[i, :, :, :3]  # Assuming first 3 channels are RGB
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        plt.imshow(img)
        plt.title(f'Image {i + 1}')
        plt.axis('off')
    plt.show()

# Directory containing the .h5 files
directory = '/kaggle/input/dataset-compressed/Sagar'

data_array = read_all_files(directory, max_files=36)
display_sample_images(data_array, num_samples=36)

# 5-fold cross validation and training/testing the CNN

epochs = 2000

for fold, (train_index, test_index) in enumerate(kf.split(x_data)):
    print(f'Fold {fold + 1}')
    
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train_loss_history = []
    test_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(epochs):
        train_loss = 0
        correct_train = 0
        total_train = 0
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_train += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_loss_history.append(avg_train_loss)
        train_accuracy_history.append(train_accuracy)
        
        model.eval()
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = 100 * correct / total
        
        test_loss_history.append(avg_test_loss)
        test_accuracy_history.append(avg_test_accuracy)

        scheduler.step(avg_test_loss)

    fold_results['train_loss'].append(train_loss_history)
    fold_results['test_loss'].append(test_loss_history)
    fold_results['train_accuracy'].append(train_accuracy_history)
    fold_results['test_accuracy'].append(test_accuracy_history)

    print(f"Fold {fold + 1} - Max Training Accuracy: {max(train_accuracy_history):.2f}%, Max Test Accuracy: {max(test_accuracy_history):.2f}%")
    print(f"Fold {fold + 1} - Min Training Loss: {min(train_loss_history):.3f}, Min Test Loss: {min(test_loss_history):.3f}")

# Averaging the results over all folds
avg_train_loss = np.mean([np.mean(fold_results['train_loss'][i]) for i in range(5)])
avg_test_loss = np.mean([np.mean(fold_results['test_loss'][i]) for i in range(5)])
avg_train_accuracy = np.mean([np.mean(fold_results['train_accuracy'][i]) for i in range(5)])
avg_test_accuracy = np.mean([np.mean(fold_results['test_accuracy'][i]) for i in range(5)])

print(f"Average Training Loss: {avg_train_loss:.3f}, Average Test Loss: {avg_test_loss:.3f}")
print(f"Average Training Accuracy: {avg_train_accuracy:.2f}%, Average Test Accuracy: {avg_test_accuracy:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# Plot training loss history
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()

# Plot test loss history
plt.subplot(2, 2, 2)
plt.plot(test_loss_history, label='Test Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss History')
plt.legend()

# Plot training accuracy history
plt.subplot(2, 2, 3)
plt.plot(train_accuracy_history, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy History')
plt.legend()

# Plot test accuracy history
plt.subplot(2, 2, 4)
plt.plot(test_accuracy_history, label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy History')
plt.legend()

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

# Function to get all predictions and labels for the test dataset
def get_all_predictions(model, loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    all_probabilities = []  # Store predicted probabilities
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            all_probabilities.extend(probabilities.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probabilities)

# Get true labels and predicted labels for the test dataset
true_labels, predicted_labels, predicted_probabilities = get_all_predictions(model, test_loader)

# Debugging: Print unique values and their counts in true and predicted labels
print("Unique values in true labels and their counts:", np.unique(true_labels, return_counts=True))
print("Unique values in predicted labels and their counts:", np.unique(predicted_labels, return_counts=True), "\n")

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)
print()

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Minor', 'Moderate', 'Severe'])
fig, ax = plt.subplots(figsize=(10, 6))
disp.plot(cmap=plt.cm.Greens, ax=ax)

# Overlay a blank grid with thin black borders
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Minor', 'Moderate', 'Severe'])
ax.set_yticklabels(['Minor', 'Moderate', 'Severe'])
ax.set_xlabel('Predicted Label (0, 1, 2)')
ax.set_ylabel('True Label (0, 1, 2)')
ax.set_title('Confusion Matrix')
plt.show()

# Calculate precision and recall for each class
precision = []
recall = []

for i in range(len(cm)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision_i = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_i = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    precision.append(precision_i)
    recall.append(recall_i)

# Calculate average precision and recall
average_precision = np.mean(precision)
average_recall = np.mean(recall)

# Print results
print(f"\nPrecision for each class: {precision}")
print(f"Recall for each class: {recall}")
print(f"Average Precision: {average_precision:.3f}")
print(f"Average Recall: {average_recall:.3f}")

# Calculate F1 score for each class
f1_scores = []
for p, r in zip(precision, recall):
    f1 = 2 * (p * r) / (p + r) if (p + r) != 0 else 0
    f1_scores.append(f1)

# Calculate average F1 score
average_f1_score = np.mean(f1_scores)

print(f"\nF1 Score for each class: {f1_scores}")
print(f"Average F1 Score: {average_f1_score:.4f}")
print()

# Calculate ROC AUC
n_classes = 3
true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], predicted_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_binarized.ravel(), predicted_probabilities.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves
plt.figure(figsize=(12, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
