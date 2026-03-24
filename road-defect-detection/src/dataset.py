import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_samples(root_dir):
    classes = {"normal": 0, "pothole": 1, "crack": 1}
    samples = []

    for cls, label in classes.items():
        class_dir = os.path.join(root_dir, cls)
        if not os.path.exists(class_dir):
            continue

        for img in os.listdir(class_dir):
            if img.lower().endswith(IMG_EXTENSIONS):
                samples.append((os.path.join(class_dir, img), label))

    return samples


class RoadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = load_samples(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
