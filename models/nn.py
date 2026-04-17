import json
import os
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def load_file(file_path='data/chords/labels.json'):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Provided file path not found: {file_path}")
    with file_path.open('r') as f:
        return json.load(f)


# Recover data from JSON files
label_map = load_file('data/chords/labels.json')
chord_map = load_file('data/chords/chords.json')
landmark_map = load_file('data/chords/landmarks.json')


# Convert the landmark_map and label_map into a dataset format for training the neural network
class LandmarkDataset(Dataset):
    def __init__(self, json_path, label_map=None):
        with open(json_path, "r") as f:
            self.data_dict = json.load(f)

        self.samples = []
        self.label_map = label_map or {}

        # Build label mapping if not provided
        if not self.label_map:
            labels = [self._extract_label(fp) for fp in self.data_dict.keys()]
            unique_labels = sorted(set(labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}

        for filepath, landmarks in self.data_dict.items():
            label_name = self._extract_label(filepath)
            label = self.label_map[label_name]

            # Flatten landmarks: (21,3) → (63,) 
            # Originally in (21, 3) format b/c there are 21 landmarks with x,y,z coordinates
            features = [coord for lm in landmarks for coord in lm]

            self.samples.append((features, label))

    def _extract_label(self, filepath):
        return label_map.get(Path(filepath).name, 'UNKNOWN')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

    def get_class_names(self):
        # maps labels to their human readable names, used in graphing / visuals
        res = [
            chord_map[str(name)]
            for name, _ in sorted(self.label_map.items(), key=lambda item: item[1])
        ]        
        return res


dataset = LandmarkDataset("data/chords/landmarks.json")

# Visualize a couple metrics to make sure it was created correctly 
print(f"Dataset size: {len(dataset)}")
print(dataset.__getitem__(0))
print(f"Label mapping: {dataset.label_map}")

# Get size metrics 
num_classes = len(dataset.label_map)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
label_counts = Counter([label for _, label in dataset.samples]) # Class balance counts 
print(f"Label counts: {label_counts}")
total = sum(label_counts.values())
print("Total samples: ", total)


# Split dataset and create dataloaders 
training_dataset, testing_dataset = random_split(dataset, [train_size, test_size])
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=32)

# Create neural network to use with landmarking data 
class LandmarkNN(nn.Module):
    # Input 63 features (21 landmarks, 3 coordinates each)
    # Output num_classes dimension
    def __init__(self, input_dim=63, num_classes=num_classes, dropout=0.4):
        super(LandmarkNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LandmarkNN(input_dim=63, num_classes=num_classes).to(device)

weights = torch.tensor(
    [total / (num_classes * label_counts[i]) for i in range(num_classes)],
    dtype=torch.float,
).to(device)

loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

def save_run(accuracy, epochs, train_losses, test_losses, test_accuracies, class_names, train_labels, train_preds, test_labels, test_preds):
    folder = f'model_{accuracy:.1f}_{epochs}'
    os.makedirs(folder, exist_ok=True)
    print(class_names[0])
    


    torch.save(model.state_dict(), os.path.join(folder, 'model.pth'))

    ep = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    ax1.plot(ep, train_losses, 'b-o', label='Train Loss')
    ax1.plot(ep, test_losses, 'r-o', label='Test Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(ep, test_accuracies, 'g-o', label='Test Accuracy')
    ax2.set_title('Test Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'training_metrics.png'), dpi=150)
    plt.close()

    for labels, preds, title in [
        (train_labels, train_preds, 'Training'),
        (test_labels, test_preds, 'Testing'),
    ]:
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{title} Confusion Matrix', fontsize=14, fontweight='bold')
        for ax, data, fmt, subtitle in zip(
            axes,
            [cm, cm_normalized],
            ['d', '.1f'],
            ['Raw Counts', 'Normalized (%)'],
        ):
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(subtitle)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f'{title.lower()}_confusion_matrix.png'), dpi=150)
        plt.close()

    print(f'All outputs saved to: {folder}/')


def collect_predictions(loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_labels), np.array(all_predictions)


def get_class_names():
    return dataset.get_class_names()


def train(training_loader, testing_loader, epochs=10):
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 99:
                print(f'Epoch {epoch+1} | Step {i+1} | Loss: {epoch_loss / (i+1):.3f}')

        train_loss = epoch_loss / len(training_loader)
        train_losses.append(train_loss)

        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in testing_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(testing_loader)
        accuracy = 100 * correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        scheduler.step(avg_test_loss)

        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Test Loss: {avg_test_loss:.3f} | Accuracy: {accuracy:.2f}%')

    print('Finished Training')
    return train_losses, test_losses, test_accuracies


def test(testing_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testing_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test set: {accuracy:.2f}%')
    return accuracy


EPOCHS = 20
train_losses, test_losses, test_accuracies = train(training_dataloader, testing_dataloader, epochs=EPOCHS)
accuracy = test(testing_dataloader)

class_names = get_class_names()


train_labels, train_preds = collect_predictions(training_dataloader)
test_labels, test_preds = collect_predictions(testing_dataloader)
save_run(accuracy, EPOCHS, train_losses, test_losses, test_accuracies, class_names, train_labels, train_preds, test_labels, test_preds)


