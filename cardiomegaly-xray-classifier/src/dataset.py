import os
from glob import glob
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


def build_image_index(image_root: str) -> Dict[str, str]:
    """
    Recursively maps image filename -> full path.
    NIH images may be split across multiple folders.
    """
    image_paths = glob(os.path.join(image_root, "**", "*.png"), recursive=True)
    index = {os.path.basename(path): path for path in image_paths}
    return index


def prepare_nih_dataframe(csv_path: str, image_root: str) -> pd.DataFrame:
    """
    Loads NIH ChestX-ray14 metadata and creates a binary target:
    target = 1 if 'Cardiomegaly' appears in Finding Labels, else 0.

    Keeps only rows whose images exist on disk.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["Image Index", "Finding Labels", "Patient ID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    image_index = build_image_index(image_root)
    df["image_path"] = df["Image Index"].map(image_index)
    df = df.dropna(subset=["image_path"]).copy()

    df["target"] = df["Finding Labels"].apply(
        lambda x: 1 if "Cardiomegaly" in str(x).split("|") else 0
    )

    return df


def patient_wise_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by patient ID to avoid data leakage.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    groups = df["Patient ID"]

    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_groups = temp_df["Patient ID"]
    relative_val_size = val_size / (val_size + test_size)

    gss2 = GroupShuffleSplit(
        n_splits=1, train_size=relative_val_size, random_state=random_state
    )
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


class NIHCardiomegalyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["target"])

        if self.transform:
            image = self.transform(image)

        return image, label
