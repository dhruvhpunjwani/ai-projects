import cv2
import face_recognition
import numpy as np

from mask_utils import apply_synthetic_mask
from utils import load_encodings


ENCODINGS_FILE = "../models/face_encodings.pkl"


def main():
    known_encodings, known_names = load_encodings(ENCODINGS_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Starting masked identity detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=0.5
            )
            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )

            name = "Unknown"
            confidence = 0.0

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = max(0.0, 1.0 - float(face_distances[best_match_index]))

                if matches[best_match_index]:
                    name = known_names[best_match_index]

            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            frame = apply_synthetic_mask(frame, (top, right, bottom, left))

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({confidence:.2f})"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (left + 5, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("Mask Identity Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
