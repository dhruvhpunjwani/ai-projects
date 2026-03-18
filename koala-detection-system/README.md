# Real-Time Koala Detection System

Computer vision project using **PyTorch**, **OpenCV**, and **transfer learning** to classify koalas in images and run real-time inference on webcam or video input.

## Project Overview

This project builds a wildlife monitoring pipeline that:

- trains a binary image classifier (**koala / not_koala**)
- applies image preprocessing and augmentation
- performs real-time inference using OpenCV
- overlays live predictions on webcam or video frames

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn



## Project Structure
```text
koala-detection-system/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
└── examples/
    └── sample_frame.jpg
```

Local dataset format:
## Dataset Structure
data/
├── train/
│   ├── koala/
│   └── not_koala/
├── val/
│   ├── koala/
│   └── not_koala/
└── test/
    ├── koala/
    └── not_koala/
```


