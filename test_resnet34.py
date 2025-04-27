# test_resnet34.py
import os
import torch
import pandas as pd
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------- 1. 환경 설정 ----------
BATCH_SIZE = 32
IMG_SIZE = 224
DATA_DIR = "/home/kangkr1002/facial_bone"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet34_facial_bone.pth"

# ---------- 2. 데이터셋 ----------
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dir = os.path.join(DATA_DIR, "Test")
test_ds = ImageFolder(test_dir, transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

# ---------- 3. 모델 ----------
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# ---------- 4. 테스트 ----------
preds = []
labels = []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Predict"):
        x = x.to(DEVICE)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.tolist())

# 🔥 Normal/Abnormal 뒤집기
pred_labels = ["Normal" if p else "Abnormal" for p in preds]
true_labels = ["Normal" if l else "Abnormal" for l in labels]

# 🔥 Accuracy 계산
test_acc = sum([p == l for p, l in zip(pred_labels, true_labels)]) / len(labels)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}% ({sum([p==l for p,l in zip(pred_labels,true_labels)])}/{len(labels)})")

# 🔥 결과 저장
df = pd.DataFrame({
    "filepath": [str(p) for p, _ in test_ds.samples],
    "pred": pred_labels
})
df.to_csv("test_predictions.csv", index=False)
print("🔎 Saved test_predictions.csv")
