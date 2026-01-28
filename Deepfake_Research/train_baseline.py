import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import os
import json
from tqdm import tqdm

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
SAVE_DIR = 'checkpoints'
LOG_DIR = 'logs'

# --- DATASET ---
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Real (Label 0)
        for root, _, files in os.walk(real_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(0)
        
        # Load Fake (Label 1)
        for root, _, files in os.walk(fake_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(1)

    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
        except:
            return torch.zeros((3,224,224)), self.labels[idx]

# --- MODEL ---
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=True)
        # Adjust classifier for Xception
        if hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'get_classifier'): self.backbone.reset_classifier(0)
        
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))

    def forward(self, x):
        return self.classifier(self.backbone(x))

# --- TRAINING FUNCTION ---
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Transform
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Data
    full_ds = DeepfakeDataset('data/real', 'data/fake', tf)
    train_size = int(0.8 * len(full_ds))
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize
    model = DeepfakeDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    print("Starting Baseline Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outs = model(imgs)
                _, preds = torch.max(outs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_acc = 100 * correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Val Acc {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, f"{SAVE_DIR}/best_baseline.pth")

    # Save logs
    with open(f"{LOG_DIR}/baseline_history.json", 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    train()