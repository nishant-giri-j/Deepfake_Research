import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = 'results'

# --- REUSE CLASSES (To load weights properly) ---
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=False)
        if hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'get_classifier'): self.backbone.reset_classifier(0)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
    def forward(self, x): return self.classifier(self.backbone(x))

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for root, _, files in os.walk(real_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(0)
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
        except: return torch.zeros((3,224,224)), self.labels[idx]

# --- ATTACKS ---
def pgd_attack(model, images, labels, epsilon, alpha=0.01, iterations=10):
    original = images.clone().detach()
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1).detach()
    for _ in range(iterations):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        images = images.detach() + alpha * data_grad.sign()
        perturbation = torch.clamp(images - original, -epsilon, epsilon)
        images = torch.clamp(original + perturbation, 0, 1).detach()
    return images

def test_model(model, loader, epsilon):
    model.eval()
    preds, actuals = [], []
    for imgs, labels in tqdm(loader, desc=f"Testing eps={epsilon}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        if epsilon > 0:
            imgs = pgd_attack(model, imgs, labels, epsilon)
            
        with torch.no_grad():
            outs = model(imgs)
            _, p = torch.max(outs, 1)
            preds.extend(p.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
    return accuracy_score(actuals, preds), preds, actuals

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load Data
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_ds = DeepfakeDataset('data/real', 'data/fake', tf)
    # Use 20% for testing
    _, test_ds = torch.utils.data.random_split(full_ds, [int(0.8*len(full_ds)), len(full_ds)-int(0.8*len(full_ds))])
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Load Models
    baseline = DeepfakeDetector().to(DEVICE)
    baseline.load_state_dict(torch.load('checkpoints/best_baseline.pth', map_location=DEVICE)['model_state_dict'])
    
    robust = DeepfakeDetector().to(DEVICE)
    robust.load_state_dict(torch.load('checkpoints/best_robust.pth', map_location=DEVICE)['model_state_dict'])
    
    # Run Tests
    epsilons = [0.0, 0.01, 0.03, 0.05]
    base_scores, rob_scores = [], []
    
    # For Confusion Matrix (at 0.03)
    cm_base, cm_rob = None, None
    labels_base, labels_rob = None, None
    
    for eps in epsilons:
        acc_b, p_b, l_b = test_model(baseline, loader, eps)
        acc_r, p_r, l_r = test_model(robust, loader, eps)
        
        base_scores.append(acc_b * 100)
        rob_scores.append(acc_r * 100)
        
        if eps == 0.03:
            cm_base, labels_base = p_b, l_b
            cm_rob, labels_rob = p_r, l_r

    # 1. Plot Line Graph
    plt.figure()
    plt.plot(epsilons, base_scores, 'r-o', label='Baseline')
    plt.plot(epsilons, rob_scores, 'g-s', label='Robust')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Attack Comparison')
    plt.savefig(f"{RESULTS_DIR}/attack_comparison.png")
    
    # 2. Confusion Matrices
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    sns.heatmap(confusion_matrix(labels_base, cm_base), annot=True, fmt='d', cmap='Reds', ax=ax[0])
    ax[0].set_title("Baseline (Eps=0.03)")
    sns.heatmap(confusion_matrix(labels_rob, cm_rob), annot=True, fmt='d', cmap='Greens', ax=ax[1])
    ax[1].set_title("Robust (Eps=0.03)")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png")
    
    # 3. CSV
    df = pd.DataFrame({'Epsilon': epsilons, 'Baseline Acc': base_scores, 'Robust Acc': rob_scores})
    df.to_csv(f"{RESULTS_DIR}/results_table.csv", index=False)
    
    print("Evaluation Complete. Check 'results' folder.")