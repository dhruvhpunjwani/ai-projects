# Road Defect Detection System

Computer vision project using **PyTorch**, **OpenCV**, and **transfer learning** to classify road surfaces as **normal** or **defective** (potholes / cracks).

## Project Overview

This project builds a vision pipeline for infrastructure monitoring by:

- training a binary classifier for **road defect detection**
- applying image preprocessing and augmentation
- supporting real-time inference with OpenCV
- enabling practical use cases in smart city and transport monitoring

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn

## Dataset Structure

Expected local dataset format:
```text
data/
├── train/
│   ├── normal/
│   ├── pothole/
│   └── crack/
├── val/
│   ├── normal/
│   ├── pothole/
│   └── crack/
└── test/
    ├── normal/
    ├── pothole/
    └── crack/
```

## Project Structure
```text
road-defect-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
└── examples/
    └── sample_road.jpg
```

## Installation
```text
pip install -r requirements.txt
```

##Training
```text
cd src
python train.py
```

#Real-Time Inference
```text
cd src
python inference.py
```
