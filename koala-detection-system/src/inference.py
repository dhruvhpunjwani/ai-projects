import argparse
import os

import cv2
import torch
from torchvision import transforms
from PIL import Image

from model import build_model


CLASS_NAMES = ["Not Koala", "Koala"]


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time koala detection")
    parser.add_argument("--model_path", type=str, default="../models/best_model.pt")
    parser.add_argument("--source", type=str, default="0",
                        help="0 for webcam, or path to video file")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.6)
    return parser.parse_args()


def preprocess_frame(frame, img_size):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return tfms(image).unsqueeze(0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam/video source.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x = preprocess_frame(frame, args.img_size).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        label = CLASS_NAMES[pred_idx]
        if confidence < args.threshold:
            label = "Uncertain"

        text = f"{label}: {confidence:.2f}"
        color = (0, 255, 0) if pred_idx == 1 else (0, 0, 255)

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Koala Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
