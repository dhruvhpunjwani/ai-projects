import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import prepare_nih_dataframe, patient_wise_split, NIHCardiomegalyDataset
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train NIH Cardiomegaly Classifier")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--image_root", type=str, required=True, help="Root folder containing NIH images")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="models")
    return parser.parse_args()


def get_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def make_weighted_sampler(labels):
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
    }
    return metrics


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
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Preparing dataset...")
    df = prepare_nih_dataframe(args.csv_path, args.image_root)
    train_df, val_df, test_df = patient_wise_split(df)

    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Train class distribution:", dict(Counter(train_df["target"])))

    train_tfms, eval_tfms = get_transforms(args.img_size)

    train_dataset = NIHCardiomegalyDataset(train_df, transform=train_tfms)
    val_dataset = NIHCardiomegalyDataset(val_df, transform=eval_tfms)
    test_dataset = NIHCardiomegalyDataset(test_df, transform=eval_tfms)

    sampler = make_weighted_sampler(train_df["target"].tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: {json.dumps(val_metrics, indent=2)}")

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_model_path = os.path.join(args.model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("Test Metrics:")
    print(json.dumps(test_metrics, indent=2))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "history": history,
                "test_metrics": test_metrics,
            },
            f,
            indent=2
        )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "csv_path": args.csv_path,
                "image_root": args.image_root,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "img_size": args.img_size,
                "task": "Binary cardiomegaly detection from NIH ChestX-ray14",
                "classes": ["No Cardiomegaly", "Cardiomegaly"]
            },
            f,
            indent=2
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
