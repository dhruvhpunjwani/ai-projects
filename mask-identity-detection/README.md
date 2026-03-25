# Mask Identity Detection

Computer vision project using **OpenCV** and **face recognition** to identify individuals in real time under partial facial occlusion.

## Project Overview

This project simulates a masked-face identity recognition workflow that:

- encodes known face images for each identity
- performs real-time face detection and recognition from webcam input
- applies a synthetic lower-face mask to demonstrate partial occlusion
- displays predicted identity labels with confidence scores

## Technologies Used

- Python
- OpenCV
- face_recognition
- NumPy
- Pillow

## Expected Local Structure

```text
mask-identity-detection/
├── known_faces/
│   ├── Alice/
│   │   └── alice1.jpg
│   └── Bob/
│       └── bob1.jpg
├── models/
├── src/
│   ├── encode_faces.py
│   ├── detect_masked_identity.py
│   ├── mask_utils.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation
```text
pip install -r requirements.txt
```

## Encode Known Faces
```text
cd src
python encode_faces.py
```

## Run Real-Time Detection
```text
cd src
python detect_masked_identity.py
```

## Notes
This project uses face encodings generated from known face images.
A synthetic lower-face mask is applied during display to simulate masked identity detection.
This project is intended for educational and portfolio purposes.
