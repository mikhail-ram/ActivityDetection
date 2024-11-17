import os
import cv2
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image

class UCF50Dataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=50, num_frames=20):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.videos, self.labels, self.class_names = self._load_data()

        print(f"Selected Classes ({self.num_classes}): {self.class_names}")

    def _load_data(self):
        videos, labels = [], []
        classes = sorted(os.listdir(self.root_dir))[:self.num_classes]
        class_names = [cls_name for cls_name in classes]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            class_path = os.path.join(self.root_dir, cls_name)
            for video in os.listdir(class_path):
                videos.append(os.path.join(class_path, video))
                labels.append(self.class_to_idx[cls_name])

        return videos, labels, class_names

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path, self.num_frames)
        frames = torch.stack([self.transform(frame) for frame in frames])
        return frames, label

    def _extract_frames(self, path, num_frames):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        while len(frames) < num_frames:
            frames.append(frames[-1])

        cap.release()
        return frames

class MobileNetGRU(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MobileNetGRU, self).__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT
        self.mobilenet = mobilenet_v2(weights=weights).features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.mobilenet(x)
        features = self.avgpool(features).view(batch_size, seq_len, -1)
        out, _ = self.gru(features)
        out = self.fc(out[:, -1, :])
        return out

num_classes = 50
hidden_size = 256
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
transform = Compose([Resize((224, 224)), ToTensor()])

dataset = UCF50Dataset(root_dir='/content/UCF50/', transform=transform, num_classes=num_classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetGRU(hidden_size=hidden_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for frames, labels in tqdm(train_loader):
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {test_accuracy * 100:.4f}%')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), [acc * 100 for acc in test_accuracies], marker='o', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Curve')
plt.legend()

plt.show()

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=dataset.class_names))

torch.save(model, "/content/gru_model_complete.pth")

