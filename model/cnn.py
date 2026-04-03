from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import matplotlib.pyplot as plt
from collections import Counter
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from torch import nn
import os

ds = load_dataset("dduka/guitar-chords")

label_counts = Counter(ds['train']['label'])
num_classes  = len(label_counts)
total        = sum(label_counts.values())

print(label_counts)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def apply_transform(batch):
    batch["image"] = [transform(img.convert("RGB")) for img in batch["image"]]
    return batch

ds['train'].set_transform(apply_transform)
ds['test'].set_transform(apply_transform)

def collate_fn(examples):
    images = torch.stack([example["image"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"image": images, "label": labels}

training_dataloader = DataLoader(ds['train'], batch_size=32, collate_fn=collate_fn, shuffle=True)
testing_dataloader  = DataLoader(ds['test'], batch_size=32, collate_fn=collate_fn)


class ConvolutionNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=14, dropout=0.4):
        super(ConvolutionNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(4*4*256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionNN().to(device)

weights = torch.tensor(
    [total / (num_classes * label_counts[i]) for i in range(num_classes)],
    dtype=torch.float
).to(device)

print(weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)


def save_run(accuracy, epochs, train_losses, test_losses, test_accuracies, class_names, train_labels, train_preds, test_labels, test_preds):
    """Save all outputs for a run into a named folder."""
    folder = f"model_{accuracy:.1f}_{epochs}"
    os.makedirs(folder, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(folder, 'model.pth'))

    # Save metrics plot
    ep = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    ax1.plot(ep, train_losses, 'b-o', label='Train Loss')
    ax1.plot(ep, test_losses,  'r-o', label='Test Loss')
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

    # Save confusion matrices
    for labels, preds, title in [(train_labels, train_preds, 'Training'), (test_labels, test_preds, 'Testing')]:
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
        for batch in loader:
            inputs = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_labels), np.array(all_predictions)


def get_class_names(loader):
    class_names = ds['train'].features['label'].names if hasattr(ds['train'].features['label'], 'names') else None
    if class_names is None:
        labels, _ = collect_predictions(loader)
        class_names = [str(i) for i in range(len(np.unique(labels)))]
    return class_names


def train(training_loader, testing_loader, epochs=5):
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss   = 0.0

        for i, batch in enumerate(training_loader):
            inputs = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss   += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1} | Step {i + 1} | Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        train_losses.append(epoch_loss / len(training_loader))

        model.eval()
        correct, total, epoch_test_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in testing_loader:
                inputs = batch['image'].to(device).float()
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                epoch_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = epoch_test_loss / len(testing_loader)
        accuracy      = 100 * correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        scheduler.step(avg_test_loss)

        print(f'Epoch {epoch + 1} | Train Loss: {train_losses[-1]:.3f} | Test Loss: {avg_test_loss:.3f} | Accuracy: {accuracy:.2f}%')

    print('Finished Training')
    return train_losses, test_losses, test_accuracies


def test(testing_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in testing_loader:
            inputs = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy


# ── Run ──────────────────────────────────────────────────────────────────────
EPOCHS = 20

train_losses, test_losses, test_accuracies = train(training_dataloader, testing_dataloader, epochs=EPOCHS)
accuracy = test(testing_dataloader)

class_names = get_class_names(training_dataloader)
train_labels, train_preds = collect_predictions(training_dataloader)
test_labels,  test_preds  = collect_predictions(testing_dataloader)

save_run(accuracy, EPOCHS, train_losses, test_losses, test_accuracies,
         class_names, train_labels, train_preds, test_labels, test_preds)