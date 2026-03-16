import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predict cardiomegaly from a chest X-ray")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image_path).convert("RGB")
    x = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_names = ["No Cardiomegaly", "Cardiomegaly"]
    pred_idx = int(probs.argmax())

    result = {
        "image_path": os.path.abspath(args.image_path),
        "predicted_class": class_names[pred_idx],
        "probabilities": {
            class_names[0]: float(probs[0]),
            class_names[1]: float(probs[1]),
        }
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
