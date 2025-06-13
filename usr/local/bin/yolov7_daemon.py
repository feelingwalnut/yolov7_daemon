#!/usr/bin/env python3
import os
import socket
import pickle
from configparser import ConfigParser
from pathlib import Path
import traceback
import threading
import queue
import time

import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

# --- COCO class names ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def class_color(name):
    np.random.seed(abs(hash(name)) % 2**32)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))

# --- Letterbox Resize ---
def letterbox(im, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

# --- Inference and Annotation ---
def process_image(image_path, threshold):
    img0 = cv2.imread(image_path)
    if img0 is None:
        print(f"[WARN] Could not read image: {image_path}")
        return {"error": "Could not read image", "detections": []}

    img_resized, ratio, (dw, dh) = letterbox(img0, new_shape=imgsz, auto=False)
    img = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, threshold, 0.45, agnostic=False)

    results = []
    annotated = img0.copy()

    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                class_id = int(cls)
                confidence = float(conf)
                if allowed_classes is None or class_id in allowed_classes:
                    xyxy = torch.tensor(xyxy).cpu()
                    x1 = int((xyxy[0] - dw) / ratio)
                    y1 = int((xyxy[1] - dh) / ratio)
                    x2 = int((xyxy[2] - dw) / ratio)
                    y2 = int((xyxy[3] - dh) / ratio)

                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class{class_id}"
                    print(f"[DEBUG] {class_name}: ({x1}, {y1}) → ({x2}, {y2}) @ {confidence:.2f}")

                    results.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class": class_name
                    })

                    color = class_color(class_name)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2, cv2.LINE_AA)

    if results:
        output_path = str(Path(image_path).with_name(Path(image_path).stem + "_detected" + Path(image_path).suffix))
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated image to: {output_path}")

    return {"detections": results}

def handle_client(conn, addr):
    try:
        conn.settimeout(30)
        data = b""

        while True:
            chunk = conn.recv(8192)
            if not chunk:
                break
            data += chunk
            try:
                request = pickle.loads(data)
                break
            except (pickle.UnpicklingError, EOFError):
                continue

        if not data:
            raise RuntimeError("No data received from client")

        # Extract directory and metadata
        directory_name = request.get("directory")
        meta = request.get("meta", {})
        threshold = float(meta.get("confidence_threshold", default_threshold))

        # Find .webp image
        image_path = find_supported_image_in_directory(directory_name)
        if not image_path:
            msg = f"No .webp image found in directory: {directory_name}"
            print(f"[WARN] {msg}")
            raise FileNotFoundError(msg)

        print(f"[INFO] Found image: {image_path}")
        print(f"[REQ] Processing: {image_path} @ threshold {threshold}")
        response = process_image(str(image_path), threshold)
        response["meta"] = meta

        conn.sendall(pickle.dumps(response))

    except Exception as e:
        print(f"[ERR] Exception handling client: {e}")
        traceback.print_exc()
        try:
            conn.sendall(pickle.dumps({"error": str(e), "detections": []}))
        except Exception:
            pass
    finally:
        conn.close()

def find_supported_image_in_directory(directory):
    supported_exts = [".webp", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return None
    for ext in supported_exts:
        for file in directory_path.glob(f"*{ext}"):
            if file.is_file():
                return file
    return None

# --- Worker Thread Function ---
def worker_loop():
    while True:
        try:
            conn, addr = job_queue.get()
            try:
                handle_client(conn, addr)
            except Exception as e:
                print(f"[ERR] Worker error: {e}")
                traceback.print_exc()
                try:
                    conn.sendall(pickle.dumps({"error": "Internal server error", "detections": []}))
                except Exception:
                    pass
                finally:
                    conn.close()
            finally:
                job_queue.task_done()
        except Exception as e:
            print(f"[ERR] Exception in worker loop: {e}")
            traceback.print_exc()

# --- Acceptor Thread Function ---
def accept_connections(server_socket):
    while True:
        try:
            conn, addr = server_socket.accept()
            try:
                job_queue.put_nowait((conn, addr))
            except queue.Full:
                print("[WARN] Job queue full. Rejecting request.")
                try:
                    conn.sendall(pickle.dumps({"error": "Server busy. Try again later.", "detections": []}))
                except Exception:
                    pass
                finally:
                    conn.close()
        except Exception as e:
            print(f"[ERR] Exception in accept_connections: {e}")
            traceback.print_exc()
            # Sleep briefly to avoid tight loop on persistent errors
            time.sleep(1)

# --- Main Function ---
def main():
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    server.listen()
    print(f"[READY] YOLOv7 Daemon listening at {SOCKET_PATH}")

    # Start acceptor thread (daemon=True so it exits with main thread)
    threading.Thread(target=accept_connections, args=(server,), daemon=True).start()

    # Start worker threads
    for i in range(max_threads):
        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        print(f"[INFO] Worker thread {i+1}/{max_threads} started.")

    # Keep main thread alive
    threading.Event().wait()

# --- Script Entry ---
if __name__ == "__main__":
    try:
        print("[BOOT] Starting YOLOv7 Daemon initialization...")

        CONFIG_PATH = "/home/motion/yolov7/yolov7_daemon.conf"
        config = ConfigParser()
        config.read(CONFIG_PATH)

        weights_path = config.get("daemon", "weights_path", fallback="/home/motion/yolov7/yolov7-tiny.pt")
        default_threshold = config.getfloat("daemon", "confidence_threshold", fallback=0.40)
        class_str = config.get("daemon", "classes", fallback="")
        allowed_classes = list(map(int, class_str.split(","))) if class_str else None
        max_threads = config.getint("daemon", "max_threads", fallback=4)
        queue_maxsize = config.getint("daemon", "queue_maxsize", fallback=100)
        imgsz = config.getint("daemon", "imgsz", fallback=640)

        SOCKET_PATH = "/tmp/yolov7.sock"
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        device = select_device('')
        model = attempt_load(weights_path, map_location=device)
        model.eval()
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        print(f"[INFO] Model loaded: {weights_path}")
        print(f"[INFO] Allowed classes: {allowed_classes}")
        print(f"[INFO] Confidence threshold: {default_threshold}")
        print(f"[INFO] Image size: {imgsz}")
        print(f"[INFO] Max threads: {max_threads}, Queue size: {queue_maxsize}")

        job_queue = queue.Queue(maxsize=queue_maxsize)

        main()

    except Exception as e:
        print("[FATAL] Exception during startup:")
        print(e)
        traceback.print_exc()
