import cv2
import torch
from torchvision import transforms
from PIL import Image

from model import build_model


model = build_model()
model.load_state_dict(torch.load("../models/best_model.pt"))
model.eval()

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = tfms(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    label = "DEFECT" if pred == 1 else "NORMAL"

    color = (0, 0, 255) if pred == 1 else (0, 255, 0)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Road Defect Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
