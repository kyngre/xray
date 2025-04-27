# train_resnet34.py
import os, time, copy, torch, torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ---------- 1. 하이퍼파라미터 ----------
BATCH_SIZE = 32
EPOCHS      = 10
LR          = 3e-4
IMG_SIZE    = 224
DATA_DIR    = "/home/kangkr1002/facial_bone"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. 데이터셋 ----------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

train_ds = ImageFolder(os.path.join(DATA_DIR, "Training"),   transform=train_tf)
val_ds   = ImageFolder(os.path.join(DATA_DIR, "Validation"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"Classes: {train_ds.classes} → {len(train_ds)} train / {len(val_ds)} val images")

# ---------- 3. 모델 ----------
model = models.resnet34(weights=None)          # 사전학습 가중치 사용안함
model.fc = nn.Linear(model.fc.in_features, 2)       # 2-class
model = model.to(DEVICE)

# ---------- 4. 손실함수 & 옵티마이저 ----------
class_counts = torch.bincount(torch.tensor(train_ds.targets))
pos_weight   = class_counts[0] / class_counts[1]     # Normal vs Abnormal imbalance 보정
criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ---------- 5. 학습 루프 ----------
best_acc   = 0.0
best_model = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    t0 = time.time()
    # ---- Train ----
    model.train()
    train_loss, y_true, y_pred = 0, [], []
    for x, y in tqdm(train_loader, desc=f"[{epoch+1}/{EPOCHS}] Train"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, torch.nn.functional.one_hot(y, 2).float())
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        train_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())

    train_acc  = accuracy_score(y_true, y_pred)
    train_loss = train_loss / len(train_ds)

    # ---- Validation ----
    model.eval()
    val_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Val  "):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, torch.nn.functional.one_hot(y, 2).float())

            val_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    val_acc  = accuracy_score(y_true, y_pred)
    val_loss = val_loss / len(val_ds)
    scheduler.step()

    print(f"Epoch {epoch+1:02d} | "
          f"Train loss {train_loss:.4f}, acc {train_acc:.3f} ‖ "
          f"Val loss {val_loss:.4f}, acc {val_acc:.3f} ‖ "
          f"{time.time()-t0:.1f}s")

    # ---- 모델 저장 ----
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, "resnet34_facial_bone.pth")

print(f"✅ Training done. Best val acc = {best_acc:.3f}")
