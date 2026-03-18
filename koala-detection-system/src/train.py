import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import KoalaDataset
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Koala Detection Classifier")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    return parser.parse_args()


def get_transforms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tfms, eval_tfms = get_transforms(args.img_size)

    train_dataset = KoalaDataset(os.path.join(args.data_dir, "train"), transform=train_tfms)
    val_dataset = KoalaDataset(os.path.join(args.data_dir, "val"), transform=eval_tfms)
    test_dataset = KoalaDataset(os.path.join(args.data_dir, "test"), transform=eval_tfms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics
        }
        history.append(record)

        print("Train Loss:", round(train_loss, 4))
        print("Val Metrics:", json.dumps(val_metrics, indent=2))

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
            print("Saved best model.")

    print("\nFinal test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(json.dumps(test_metrics, indent=2))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "history": history,
            "test_metrics": test_metrics
        }, f, indent=2)


if __name__ == "__main__":
    main()
