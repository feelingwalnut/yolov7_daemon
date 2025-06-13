#!/usr/bin/env python3
import os
import socket
import pickle
import shutil
from pathlib import Path
import sys
import subprocess

# CONFIGURATION
debug_no_delete = False  # Set to True to skip all deletions

output_dir = Path("/home/motion/files")
confidence_threshold = 0.60
daemon_socket = "/tmp/yolov7.sock"
event_base_dir = Path("/tmp/motion")

# --- Find .webp image in the given directory ---
def find_webp_in_directory(directory):
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() == ".webp":
            return file
    return None

# --- Begin script ---
if len(sys.argv) < 2:
    print("[ERROR] No event directory name provided.")
    sys.exit(1)

# Construct full path from Motion's input
event_name = sys.argv[1]
image_dir = event_base_dir / event_name

if not image_dir.exists() or not image_dir.is_dir():
    print(f"[ERROR] Directory does not exist: {image_dir}")
    sys.exit(1)

# Find the .webp image in the directory
image_path = find_webp_in_directory(image_dir)
if not image_path:
    print(f"[ERROR] No .webp image found in directory: {image_dir}")
    sys.exit(1)

print(f"[INFO] Found image: {image_path}")

# Send detection request to YOLOv7 daemon
print("[INFO] Sending detection request to YOLOv7 daemon...")

request = {
    "directory": str(image_dir),
    "meta": {
        "source": "motion",
        "confidence_threshold": confidence_threshold
    }
}

try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(daemon_socket)
        client.sendall(pickle.dumps(request))
        result_data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            result_data += chunk
        result = pickle.loads(result_data)
except Exception as e:
    print(f"[ERROR] Communication with YOLOv7 daemon failed: {e}")
    sys.exit(1)

# Evaluate detections
valid_detection_found = False
detections = result.get("detections", [])

for det in detections:
    try:
        conf = float(det.get("confidence", 0))
        if conf >= confidence_threshold:
            valid_detection_found = True
            break
    except (ValueError, TypeError):
        continue

if not valid_detection_found:
    print("[INFO] No valid detections above threshold found.")

    if debug_no_delete:
        print(f"[DEBUG] Skipping deletion due to debug_no_delete=True: {image_dir}")
    else:
        try:
            print(f"[INFO] Deleting folder: {image_dir}")
            shutil.rmtree(image_dir)
            print("  → Folder deleted successfully.")
        except Exception as e:
            print(f"  × Failed to delete folder: {e}")

    sys.exit(0)

# Try to find the annotated image
annotated_image = None
for ext in [".jpg", ".jpeg", ".png", ".webp"]:
    candidate = image_path.with_name(image_path.stem + "_detected" + ext)
    if candidate.exists():
        annotated_image = candidate
        break

if annotated_image:
    print(f"[INFO] Found annotated image: {annotated_image}")
else:
    print("[WARN] No annotated image found. Will use original for fallback.")
    annotated_image = image_path

# Clean up .txt detection files before moving the folder
for file in image_dir.glob("*.txt"):
    try:
        file.unlink()
        print(f"[INFO] Deleted detection txt file: {file}")
    except Exception as e:
        print(f"[WARN] Failed to delete txt file: {file} → {e}")

# Move processing directory to output location
destination = output_dir / image_dir.name

print(f"[INFO] Detection found. Moving directory: {image_dir} → {destination}")
try:
    shutil.move(str(image_dir), destination)
    print(f"  → Successfully moved to {destination}")
except Exception as e:
    print(f"  × Failed to move directory: {e}")
    sys.exit(1)

# Recursively set permissions to 777
print(f"[INFO] Setting permissions to 777 for: {destination}")
try:
    for root, dirs, files in os.walk(destination):
        os.chmod(root, 0o777)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)
    print("  → Permissions successfully updated.")
except Exception as e:
    print(f"  × Failed to change permissions: {e}")

# Send notification with annotated image (if available)
try:
    push_image_path = destination / annotated_image.name
    curl_cmd = [
        "curl", "-s",
        "--form-string", "token=APIKEY",
        "--form-string", "user=LOGININFORMATION",
        "--form-string", "message=Motion!",
        "--form", f"attachment=@{push_image_path}",
        "https://api.pushover.net/1/messages.json"
    ]
    subprocess.run(curl_cmd, check=True)
    print(f"  → Sent Pushover notification with image: {push_image_path.name}")
except Exception as e:
    print(f"  × Failed to send Pushover notification: {e}")
