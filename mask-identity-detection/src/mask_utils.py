import cv2
import numpy as np


def apply_synthetic_mask(frame, face_location):
    """
    Draws a synthetic mask over the lower half of the detected face.
    face_location format: (top, right, bottom, left)
    """
    top, right, bottom, left = face_location

    face_height = bottom - top
    mask_top = top + face_height // 2

    cv2.rectangle(
        frame,
        (left, mask_top),
        (right, bottom),
        (255, 0, 0),
        thickness=-1
    )

    cv2.putText(
        frame,
        "MASK",
        (left + 5, bottom - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame


def crop_upper_face(image, face_location):
    """
    Returns upper-half crop of the face region.
    """
    top, right, bottom, left = face_location
    face = image[top:bottom, left:right]

    if face.size == 0:
        return None

    h = face.shape[0]
    upper_face = face[: h // 2, :]
    return upper_face
