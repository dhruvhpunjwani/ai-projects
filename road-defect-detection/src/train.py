import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import json

from dataset import RoadDataset
from model import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = RoadDataset("../data/train", transform=train_tfms)
val_ds = RoadDataset("../data/val", transform=eval_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = build_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_f1 = 0

for epoch in range(5):
    model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)

            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Epoch {epoch}: Acc={acc:.3f}, F1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs("../models", exist_ok=True)
        torch.save(model.state_dict(), "../models/best_model.pt")

json.dump({"best_f1": best_f1}, open("../outputs.json", "w"))
