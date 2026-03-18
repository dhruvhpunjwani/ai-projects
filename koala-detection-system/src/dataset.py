import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_image_paths(root_dir: str) -> List[Tuple[str, int]]:
    """
    Expected structure:
    data/
      train/
        koala/
        not_koala/
      val/
        koala/
        not_koala/
      test/
        koala/
        not_koala/
    """
    classes = {"koala": 1, "not_koala": 0}
    samples = []

    for class_name, label in classes.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                samples.append((os.path.join(class_dir, fname), label))

    return samples


class KoalaDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = load_image_paths(root_dir)
        self.transform = transform

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
