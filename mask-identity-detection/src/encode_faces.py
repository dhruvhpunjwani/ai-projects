import os

import face_recognition

from utils import save_encodings


def load_known_faces(known_faces_dir: str):
    """
    Expected structure:
    known_faces/
      Alice/
        img1.jpg
      Bob/
        img1.jpg
    """
    known_encodings = []
    known_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"[INFO] Encoded {image_name} for {person_name}")
                else:
                    print(f"[WARNING] No face found in {image_path}")
            except Exception as e:
                print(f"[ERROR] Failed on {image_path}: {e}")

    return known_encodings, known_names


if __name__ == "__main__":
    known_faces_dir = "../known_faces"
    output_file = "../models/face_encodings.pkl"

    os.makedirs("../models", exist_ok=True)

    encodings, names = load_known_faces(known_faces_dir)
    save_encodings(encodings, names, output_file)
