import os
import pickle

import cv2
import face_recognition
import numpy as np

from utils import mark_attendance


ENCODINGS_FILE = "../models/face_encodings.pkl"
ATTENDANCE_FILE = "../outputs/attendance.csv"


def load_encodings(encodings_file: str):
    if not os.path.exists(encodings_file):
        raise FileNotFoundError(
            f"Encodings file not found: {encodings_file}\n"
            "Run encode_faces.py first."
        )

    with open(encodings_file, "rb") as f:
        data = pickle.load(f)

    return data["encodings"], data["names"]


def main():
    known_encodings, known_names = load_encodings(ENCODINGS_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name, ATTENDANCE_FILE)

            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("Autonomous Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
