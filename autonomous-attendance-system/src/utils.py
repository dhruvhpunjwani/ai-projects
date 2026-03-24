import csv
import os
from datetime import datetime


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def mark_attendance(name: str, attendance_file: str):
    """
    Marks attendance only once per run/day per person in the CSV.
    """
    ensure_dir(os.path.dirname(attendance_file))

    existing_names = set()

    if os.path.exists(attendance_file):
        with open(attendance_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    existing_names.add(row[0])

    if name not in existing_names:
        now = datetime.now()
        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            if os.path.getsize(attendance_file) == 0:
                writer.writerow(["Name", "Date", "Time"])
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        print(f"[INFO] Attendance marked for {name}")
