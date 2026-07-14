import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import json
import os
from pyparsing import Word, alphas, nums

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

# --- 1. DATASET DEFINITION ---
class ProduceDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map: {image_id: category_id}
        self.img_to_cat = {ann['image_id']: ann['category_id'] for ann in self.data['annotations']}
        self.images = [img for img in self.data['images'] if img['id'] in self.img_to_cat]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Label adjustment: (1,2,3) -> (0,1,2)
        label = self.img_to_cat[img_info['id']] - 1
        
        if self.transform:
            image = self.transform(image)
        return image, label, img_info['file_name'] # Returning filename for mapping

# --- 2. SETUP ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = ProduceDataset('instances_default_produce.json', 'custom_produce', transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Class Mapping for JSON
class_map = {0: "fresh", 1: "edible_soon", 2: "spoiled"}
with open('class_map.json', 'w') as f:
    json.dump(class_map, f, indent=4)

# --- 3. TRAINING LOOP WITH METRICS ---
num_epochs = 25
learning_rate=1e-4
batch=16
history = {"train_acc": [], "val_acc": [], "val_f1": []}

for epoch in range(num_epochs):
    model.train()
    train_correct, train_total = 0, 0
    
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    # Validation Phase
    model.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    t_acc = 100. * train_correct / train_total
    v_acc = 100. * val_correct / val_total
    v_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    history["val_f1"].append(v_f1)

    print(f"Epoch {epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | F1: {v_f1:.4f}")
       


# --- 4. GRAPHING ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), history["train_acc"], label="Train Accuracy")
plt.plot(range(1, num_epochs+1), history["val_acc"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("accuracy_plot.png")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), history["val_f1"], color='green', label="Validation F1 Score")
plt.title("F1 Score Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig("f1_score_plot.png")

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
ticks = range(len(class_map))
plt.xticks(ticks, list(class_map.values()))
plt.yticks(ticks, list(class_map.values()))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.title("Produce Freshness Classification")
plt.xlabel("Predicted (ResNet)")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("resnet_accuracy_heatmap.png")
plt.show()

# --- 5. SAVE PREDICTIONS JSON ---
model.eval()
final_results = []
with torch.no_grad():
    for inputs, labels, filenames in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        for i in range(len(filenames)):
            res = {
                "file_name": filenames[i],
                "prediction_idx": int(predicted[i]),
                "prediction_label": class_map[int(predicted[i])],
                "ground_truth_label": class_map[int(labels[i])]
            }
            final_results.append(res)

with open('val_predictions.json', 'w') as f:
    json.dump(final_results, f, indent=4)

torch.save(model.state_dict(), 'resnet18_produce.pth')
