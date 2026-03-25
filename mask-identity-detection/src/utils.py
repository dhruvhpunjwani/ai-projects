import os
import pickle


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_encodings(encodings, names, output_file: str):
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"[INFO] Saved encodings to {output_file}")


def load_encodings(encodings_file: str):
    if not os.path.exists(encodings_file):
        raise FileNotFoundError(
            f"Encodings file not found: {encodings_file}\nRun encode_faces.py first."
        )

    with open(encodings_file, "rb") as f:
        data = pickle.load(f)

    return data["encodings"], data["names"]
