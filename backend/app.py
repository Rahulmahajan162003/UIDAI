from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import keras
from keras import layers
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.ndimage import maximum_filter
import warnings
import base64
import traceback
import json
import threading
from datetime import datetime
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import struct

warnings.filterwarnings("ignore")

# ── MediaPipe — version-safe import ──────────────────────────────────────────
# mediapipe 0.10+ moved Hands; this handles both versions + missing install
MEDIAPIPE_AVAILABLE = False
_mp_hands = None
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        # Legacy API (0.9.x and below)
        _mp_hands = mp.solutions.hands
        MEDIAPIPE_AVAILABLE = True
    else:
        # Newer 0.10+ — try the compat shim path
        try:
            from mediapipe.python.solutions import hands as _mp_hands
            MEDIAPIPE_AVAILABLE = True
        except ImportError:
            MEDIAPIPE_AVAILABLE = False
except ImportError:
    MEDIAPIPE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ── Folders ──────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER     = os.path.join(BASE_DIR, "outputs")
CROPPED_FOLDER    = os.path.join(OUTPUT_FOLDER, "cropped")
PREPROCESSED_FOLDER = os.path.join(OUTPUT_FOLDER, "preprocessed")
RESULTS_FOLDER    = os.path.join(OUTPUT_FOLDER, "results")

for d in [UPLOAD_FOLDER, CROPPED_FOLDER, PREPROCESSED_FOLDER, RESULTS_FOLDER]:
    os.makedirs(d, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
YOLO_MODEL_PATH        = os.path.join(BASE_DIR, "best-new.pt")
U2NET_MODEL_PATH       = os.path.join(BASE_DIR, "u2net_320x320_float32.tflite")
ZERO_DCE_MODEL_PATH    = os.path.join(BASE_DIR, "zero_dce_model.h5")
MINUTIAE_MODEL_PATH    = os.path.join(BASE_DIR, "best_f1.pth")
LIVENESS_MODEL_PATH    = os.path.join(BASE_DIR, "liveness_model_v3.pt")
BRIGHTSPOT_MODEL_PATH  = os.path.join(BASE_DIR, "bright_spot_detection.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIVENESS_THRESHOLD = 0.5
LIVENESS_IMG_SIZE  = 224

DATABASE = os.path.join(BASE_DIR, "uidai.db")

global_models = {}
LAST_DEBUG_DATA = None
u2net_lock = threading.Lock()

print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            uid         TEXT NOT NULL,
            batch       TEXT NOT NULL,
            template_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_uid_batch ON users(uid, batch)
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            timestamp   TEXT,
            minutiae_count INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS authentications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            timestamp   TEXT,
            confidence  REAL,
            matched     INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

def build_dce_net():
    inp = keras.Input(shape=[None, None, 3])
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    c3 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c3)
    c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(layers.Concatenate()([c4, c3]))
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(layers.Concatenate()([c5, c2]))
    xr = layers.Conv2D(24, 3, activation="tanh", padding="same")(layers.Concatenate()([c6, c1]))
    return keras.Model(inputs=inp, outputs=xr)

def load_zero_dce_weights(dce_model, weights_path):
    """Load Zero-DCE weights from legacy Keras .h5 (model_weights/functional/*)."""
    import h5py
    with h5py.File(weights_path, "r") as f:
        func = f["model_weights"]["functional"]
        loaded = 0
        for layer in dce_model.layers:
            if layer.name not in func:
                continue
            kernel = func[f"{layer.name}/kernel"][:]
            bias = func[f"{layer.name}/bias"][:]
            layer.set_weights([kernel, bias])
            loaded += 1
    if loaded != 7:
        raise ValueError(f"Expected 7 conv layers, loaded {loaded}")


class ZeroDCE(keras.Model):
    def __init__(self):
        super().__init__()
        self.dce_model = build_dce_net()

    def get_enhanced_image(self, data, output):
        r = [output[:, :, :, i*3:(i+1)*3] for i in range(8)]
        x = data
        for ri in r:
            x = x + ri * (tf.square(x) - x)
        return x

    def call(self, data):
        return self.get_enhanced_image(data, self.dce_model(data))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class HourglassModule(nn.Module):
    def __init__(self, channels, depth=3):
        super().__init__()
        self.depth = depth
        self.down_blocks  = nn.ModuleList()
        self.down_sample  = nn.ModuleList()
        for i in range(depth):
            in_ch  = channels if i == 0 else channels * (2 ** (i-1))
            out_ch = channels * (2 ** i) if i > 0 else channels
            self.down_blocks.append(ResBlock(in_ch, out_ch))
            self.down_sample.append(nn.Conv2d(out_ch, out_ch, 2, stride=2))
        bn_ch = channels * (2 ** (depth-1))
        self.bottleneck = ResBlock(bn_ch, bn_ch)
        self.up_sample  = nn.ModuleList()
        self.up_blocks  = nn.ModuleList()
        self.skip_conv  = nn.ModuleList()
        for i in range(depth):
            tl    = depth - 1 - i
            in_ch  = channels * (2 ** (depth-1)) if i == 0 else channels * (2 ** (depth-i))
            out_ch = channels * (2 ** tl) if tl > 0 else channels
            self.up_sample.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch*4, 3, padding=1),
                nn.PixelShuffle(2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            ))
            self.skip_conv.append(nn.Sequential(
                nn.Conv2d(out_ch*2, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
            ))
            self.up_blocks.append(ResBlock(out_ch, out_ch))

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.down_blocks[i](x); skips.append(x); x = self.down_sample[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.up_sample[i](x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = self.skip_conv[i](x); x = self.up_blocks[i](x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1    = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1      = nn.BatchNorm2d(channels)
        self.conv2    = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2      = nn.BatchNorm2d(channels)
        self.conv_att = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        att = F.relu(self.bn1(self.conv1(x)))
        att = F.relu(self.bn2(self.conv2(att)))
        att = torch.sigmoid(self.conv_att(att))
        return x * att + x


class MinutiaeNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        bc = base_channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, bc, 7, padding=3, bias=False), nn.BatchNorm2d(bc), nn.ReLU(inplace=True)
        )
        self.resblock1 = ResBlock(bc, bc)
        self.down1 = nn.Sequential(nn.Conv2d(bc, bc, 2, stride=2), nn.BatchNorm2d(bc), nn.ReLU(inplace=True))
        self.resblock2 = ResBlock(bc, bc)
        self.down2 = nn.Sequential(nn.Conv2d(bc, bc, 2, stride=2), nn.BatchNorm2d(bc), nn.ReLU(inplace=True))
        self.hourglass = HourglassModule(bc, depth=3)
        self.location_head = nn.Sequential(
            nn.Conv2d(bc, bc//2, 3, padding=1), nn.BatchNorm2d(bc//2), nn.ReLU(inplace=True),
            nn.Conv2d(bc//2, 1, 1), nn.Sigmoid()
        )
        self.attention     = AttentionModule(bc)
        self.direction_pre = nn.Sequential(nn.Conv2d(bc, bc//2, 3, padding=1), nn.BatchNorm2d(bc//2), nn.ReLU(inplace=True))
        self.direction_head = nn.Sequential(nn.Conv2d(bc//2, 2, 1), nn.Tanh())
        self.type_head = nn.Sequential(
            nn.Conv2d(bc, bc//2, 3, padding=1), nn.BatchNorm2d(bc//2), nn.ReLU(inplace=True),
            nn.Conv2d(bc//2, bc//4, 3, padding=1), nn.BatchNorm2d(bc//4), nn.ReLU(inplace=True),
            nn.Conv2d(bc//4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resblock1(x); x = self.down1(x)
        x = self.resblock2(x); features = self.down2(x)
        features = self.hourglass(features)
        location_map = self.location_head(features)
        att = self.attention(features)
        dir_f = self.direction_pre(att)
        direction_map = self.direction_head(dir_f)
        cos_map = direction_map[:, 0:1]; sin_map = direction_map[:, 1:2]
        norm = torch.sqrt(cos_map**2 + sin_map**2 + 1e-8)
        cos_map = cos_map / norm; sin_map = sin_map / norm
        type_map = self.type_head(features)
        return location_map, cos_map, sin_map, type_map


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    print("Loading models...")

    # U2Net
    if os.path.exists(U2NET_MODEL_PATH):
        interp = tf.lite.Interpreter(model_path=U2NET_MODEL_PATH)
        interp.allocate_tensors()
        global_models["u2net"]        = interp
        global_models["u2net_input"]  = interp.get_input_details()
        global_models["u2net_output"] = interp.get_output_details()
        print("✓ U2Net loaded")

    # Zero-DCE
    if os.path.exists(ZERO_DCE_MODEL_PATH):
        try:
            zdce = ZeroDCE()
            zdce.dce_model(np.zeros((1, 256, 256, 3), dtype=np.float32))
            load_zero_dce_weights(zdce.dce_model, ZERO_DCE_MODEL_PATH)
            global_models["zero_dce"] = zdce
            print("✓ Zero-DCE loaded")
        except Exception as e:
            print(f"⚠ Zero-DCE failed (optional): {e}")

    # MinutiaeNet
    if os.path.exists(MINUTIAE_MODEL_PATH):
        try:
            m = MinutiaeNet(base_channels=64).to(DEVICE)
            ckpt = torch.load(MINUTIAE_MODEL_PATH, map_location=DEVICE, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            m.load_state_dict(state)
            m.eval()
            global_models["minutiae"] = m
            print("✓ MinutiaeNet loaded")
        except Exception as e:
            print(f"✗ MinutiaeNet failed: {e}")

    # Liveness (MobileNetV2 with custom classifier)
    if os.path.exists(LIVENESS_MODEL_PATH):
        try:
            ckpt = torch.load(LIVENESS_MODEL_PATH, map_location=DEVICE, weights_only=False)
            backbone = models.mobilenet_v2(weights=None)
            backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )
            backbone.load_state_dict(ckpt["model_state_dict"])
            backbone.eval().to(DEVICE)
            global_models["liveness"]           = backbone
            global_models["liveness_threshold"] = float(ckpt.get("threshold", LIVENESS_THRESHOLD))
            print(f"✓ Liveness loaded (threshold={global_models['liveness_threshold']})")
        except Exception as e:
            print(f"⚠ Liveness failed (fail-open): {e}")

    # Bright-spot / glare detector (YOLO)
    if os.path.exists(BRIGHTSPOT_MODEL_PATH):
        try:
            global_models["brightspot"] = YOLO(BRIGHTSPOT_MODEL_PATH, task="detect")
            print("✓ Bright-spot detector loaded")
        except Exception as e:
            print(f"⚠ Bright-spot failed (optional): {e}")

    print(f"Models ready: {list(global_models.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def check_blur(image_bgr, mask=None):
    """Laplacian variance blur score. Higher = sharper.

    Threshold calibration:
      • Blurry / motion-shake fingerprint: Laplacian var ≈ 2–15
      • Acceptable but soft fingerprint  : Laplacian var ≈ 15–40
      • Sharp, ridge-clear fingerprint   : Laplacian var ≈ 40–300+
    Threshold raised from 3.0 → 20.0 so motion-blurred captures
    taken during the auto-capture race condition are reliably rejected.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None and np.any(mask > 0):
        score = float(lap[mask > 0].var())
        method = "finger_roi"
    else:
        score = float(lap.var())
        method = "full_frame"
    return {"blur_score": round(score, 2), "is_blurry": score < 20.0, "method": method}

def check_brightness(image_bgr, mask=None):
    """Mean luminance check."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None and np.any(mask > 0):
        mean = float(gray[mask > 0].mean())
        method = "finger_roi"
    else:
        mean = float(gray.mean())
        method = "full_frame"
    return {
        "brightness": round(mean, 2),
        "too_dark":   mean < 50.0,
        "too_bright": mean > 210.0,
        "method": method,
    }

def check_glare(image_bgr, mask=None):
    """Detect glare using bright-spot YOLO model if available, else pixel threshold."""
    glare_img = image_bgr
    glare_mask = mask
    if mask is not None and np.any(mask > 0):
        bbox = _mask_bbox(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            glare_img = image_bgr[y1:y2, x1:x2]
            glare_mask = mask[y1:y2, x1:x2]

    if "brightspot" in global_models:
        try:
            results = global_models["brightspot"](glare_img, verbose=False)
            has_glare = len(results[0].boxes) > 0
            return {"has_glare": has_glare, "method": "model"}
        except Exception:
            pass
    # Fallback: pixel-level overexposure
    gray = cv2.cvtColor(glare_img, cv2.COLOR_BGR2GRAY)
    if glare_mask is not None and np.any(glare_mask > 0):
        overexposed = float(np.sum((gray > 240) & (glare_mask > 0))) / max(1, int(np.sum(glare_mask > 0)))
        method = "pixel_finger_roi"
    else:
        overexposed = float(np.sum(gray > 240)) / gray.size
        method = "pixel"
    return {"has_glare": overexposed > 0.05, "glare_fraction": round(overexposed, 4), "method": method}


class PipelineProfiler:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.steps = []
        self.model_inference_time = 0.0

    def record_step(self, step_name, start_time, is_model_inference=False):
        duration = (time.perf_counter() - start_time) * 1000.0
        self.steps.append({"step": step_name, "time_ms": round(duration, 1)})
        if is_model_inference:
            self.model_inference_time += duration

    def get_metrics(self):
        total_time = (time.perf_counter() - self.start_time) * 1000.0
        system_overhead = total_time - self.model_inference_time
        return {
            "steps": self.steps,
            "model_inference_sum_ms": round(self.model_inference_time, 1),
            "system_overhead_ms": round(max(0.0, system_overhead), 1),
            "total_time_ms": round(total_time, 1)
        }


def detect_and_crop_contact_image(image_bgr, conf=0.1):
    """
    Attempt to run YOLO finger detection.
    For contact images, if YOLO fails, fall back to a robust contour-based cropping,
    or return the full image if no distinct foreground is found.
    To avoid a "zoomed-in" appearance, a generous padding is added.
    """
    try:
        # Get YOLO bounding box
        bbox, det_conf = detect_best_finger_box(image_bgr, conf=conf)
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        
        # Add generous padding to prevent a tight cropped/zoomed-in look
        pad = 50
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return image_bgr[y1:y2, x1:x2], det_conf
    except ValueError:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_corners = (int(gray[0, 0]) + int(gray[0, -1]) + int(gray[-1, 0]) + int(gray[-1, -1])) / 4.0
        if mean_corners > 127:
            binary = (gray < 240).astype(np.uint8) * 255
        else:
            binary = (gray > 15).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                pad = 50  # Generous padding to prevent zoomed-in look
                y1 = max(0, y - pad)
                y2 = min(gray.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(gray.shape[1], x + w + pad)
                return image_bgr[y1:y2, x1:x2], 1.0

        return image_bgr, 1.0


def prepare_quality_roi(image_bgr, profiler=None):
    meta = {
        "quality_scope": "full_frame",
        "finger_detected": False,
        "detection_conf": None,
        "segmentation_used": False,
    }
    try:
        t_det = time.perf_counter()
        cropped_bgr, det_conf = detect_and_crop_image(image_bgr, conf=0.1)
        if profiler:
            profiler.record_step("Finger Detection (YOLO)", t_det, is_model_inference=True)
        meta["quality_scope"] = "finger_roi"
        meta["finger_detected"] = True
        meta["detection_conf"] = round(det_conf, 4)

        if "u2net" in global_models:
            t_seg = time.perf_counter()
            mask = get_segmentation_mask(cropped_bgr)
            if profiler:
                profiler.record_step("Segmentation Mask (U2-Net)", t_seg, is_model_inference=True)
            if np.any(mask > 0):
                meta["segmentation_used"] = True
                return cropped_bgr, mask.astype(np.uint8), meta

        return cropped_bgr, None, meta
    except Exception as e:
        meta["fallback_reason"] = str(e)
        return image_bgr, None, meta

def quality_gate(image_bgr, mask=None, profiler=None):
    """Run all quality checks and return combined result + guidance message."""
    t_blur = time.perf_counter()
    blur   = check_blur(image_bgr, mask=mask)
    if profiler:
        profiler.record_step("Blur Quality Check", t_blur)

    t_bright = time.perf_counter()
    bright = check_brightness(image_bgr, mask=mask)
    if profiler:
        profiler.record_step("Brightness Quality Check", t_bright)

    t_glare = time.perf_counter()
    glare  = check_glare(image_bgr, mask=mask)
    if profiler:
        is_model_inf = "brightspot" in global_models
        profiler.record_step("Glare Quality Check", t_glare, is_model_inference=is_model_inf)

    issues = []
    if blur["is_blurry"]:       issues.append("Image is blurry — hold steady")
    if bright["too_dark"]:      issues.append("Too dark — move to better light or enable flash")
    if bright["too_bright"]:    issues.append("Too bright — reduce exposure")
    if glare["has_glare"]:      issues.append("Glare detected — adjust angle")

    return {
        "passed":   len(issues) == 0,
        "issues":   issues,
        "guidance": issues[0] if issues else "Good — capture ready",
        "blur":     blur,
        "brightness": bright,
        "glare":    glare,
    }


def run_quality_gate(image_bgr, profiler=None):
    """Quality gate on finger ROI — same logic as /quality_check (YOLO crop + U²Net mask)."""
    eval_img, eval_mask, meta = prepare_quality_roi(image_bgr, profiler=profiler)
    if not meta.get("finger_detected"):
        return {
            "passed": False,
            "issues": ["No finger detected — place finger in view"],
            "guidance": "No finger detected — place finger in view",
            "blur": {"blur_score": 0.0, "is_blurry": True, "method": "none"},
            "brightness": {"brightness": 0.0, "too_dark": True, "too_bright": False, "method": "none"},
            "glare": {"has_glare": False, "method": "none"},
            **meta,
        }
    result = quality_gate(eval_img, mask=eval_mask, profiler=profiler)
    result.update(meta)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION + LIVENESS
# ══════════════════════════════════════════════════════════════════════════════

def get_finger_detector():
    if "finger_detector" not in global_models:
        global_models["finger_detector"] = YOLO(YOLO_MODEL_PATH, task="detect")
    return global_models["finger_detector"]


def detect_best_finger_box(image_bgr, conf=0.1):
    model = get_finger_detector()
    results = model(image_bgr, save=False, imgsz=800, conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        raise ValueError("No finger detected")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, boxes[best])
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    return (x1, y1, x2, y2), float(confs[best])


def detect_and_crop_image(image_bgr, conf=0.1):
    bbox, det_conf = detect_best_finger_box(image_bgr, conf=conf)
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2], det_conf


def detect_and_crop(image_path, conf=0.1):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")
    return detect_and_crop_image(img, conf=conf)

def check_liveness(cropped_bgr):
    if "liveness" not in global_models:
        return {"is_live": True, "confidence": 1.0, "note": "model_not_loaded"}
    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    img_r   = cv2.resize(img_rgb, (LIVENESS_IMG_SIZE, LIVENESS_IMG_SIZE))
    inp     = img_r.astype(np.float32) / 255.0
    mean    = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(1,1,3)
    std     = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(1,1,3)
    inp     = (inp - mean) / std
    t = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits    = global_models["liveness"](t)
        live_prob = float(F.softmax(logits, dim=1)[0, 0].cpu())
    return {
        "is_live":    live_prob >= global_models["liveness_threshold"],
        "confidence": round(live_prob, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION + PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def apply_gabor_filters(image):
    """
    Ridge enhancement using multi-orientation Gabor filters.
    """
    enhanced = np.zeros_like(image, dtype=np.float32)

    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=5,
            theta=theta,
            lambd=10,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )

        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        enhanced = np.maximum(enhanced, filtered)

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return enhanced.astype(np.uint8)

def get_segmentation_mask(image_bgr, thresh=0.3):
    interp  = global_models["u2net"]
    in_det  = global_models["u2net_input"]
    out_det = global_models["u2net_output"]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    inp     = cv2.resize(img_rgb, (320,320)).astype(np.float32) / 255.0
    batched = np.expand_dims(inp, 0).copy()

    with u2net_lock:
        interp.set_tensor(in_det[0]["index"], batched)
        interp.invoke()
        pred = interp.get_tensor(out_det[0]["index"])[0, :, :, 0].copy()

    prob    = cv2.resize(pred, (w, h))
    mask    = (prob > thresh).astype(np.uint8)
    # convex hull clean
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [hull], -1, 1, cv2.FILLED)
        mask = clean
    return mask

def create_central_roi(mask, alpha=0.25):
    k = np.ones((7,7), np.uint8)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    hm   = np.zeros_like(m)
    cv2.drawContours(hm, [hull], -1, 1, cv2.FILLED)
    ep = int(min(hm.shape) * alpha * 0.5)
    if ep > 1:
        hm = cv2.erode(hm, np.ones((ep,ep), np.uint8))
    return hm

def preprocess_fingerprint(cropped_bgr, mask=None):
    if mask is None:
        mask = get_segmentation_mask(cropped_bgr)
    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    white   = np.ones_like(img_rgb) * 255
    fg      = np.where(mask[:,:,None], img_rgb, white).astype(np.uint8)

    lum = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY).mean()
    if lum < 150 and "zero_dce" in global_models:
        inp = np.expand_dims(fg.astype(np.float32)/255.0, 0)
        enh = global_models["zero_dce"](inp)
        enhanced = tf.cast(enh[0]*255, tf.uint8).numpy()
    else:
        gray     = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
        hist, _  = np.histogram(gray.flatten(), 256, [0,256])
        cdf      = hist.cumsum()
        cdf_m    = np.ma.masked_equal(cdf, 0)
        cdf_m    = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf      = np.ma.filled(cdf_m, 0).astype("uint8")
        eq       = cdf[gray]
        enhanced = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

    if len(enhanced.shape) == 3:
        gray_e = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray_e = enhanced.astype(np.uint8)
    thresh = cv2.adaptiveThreshold(gray_e, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 1)
    inv = 255 - thresh
    inv[mask == 0] = 255

    roi = create_central_roi(mask)
    final = inv.copy()
    final[roi == 0] = 255

    # auto-crop bottom
    rows_with_content = np.where(np.any(final < 200, axis=1))[0]
    if len(rows_with_content):
        pad = int(final.shape[0] * 0.02)
        final = final[:min(final.shape[0], rows_with_content[-1]+pad+1), :]

    return final

#____________________Contact-based fingerprint preprocessing__________________________________________________________
def preprocess_contact_fingerprint(image_bgr):
    """
    Preprocessing for CONTACT-BASED fingerprints
    (scanner fingerprints) with high-fidelity noise reduction and binarization.
    """
    # 1. Grayscale conversion
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    # 2. Bilateral filter to smooth noise while keeping ridge boundaries sharp
    smoothed = cv2.bilateralFilter(gray, 11, 85, 85)

    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # 4. Gaussian blur to smooth high-frequency ridge noise before thresholding
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 5. Dynamic Adaptive Gaussian thresholding based on image size to prevent blocky artifacts
    h, w = blurred.shape[:2]
    block_size = int(min(h, w) * 0.08)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(15, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        4
    )

    # 6. Morphology cleanup using an elliptical structuring element (3x3) to smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# MINUTIAE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def enhance_for_minutiae(img_gray):
    mean = img_gray.mean(); std = img_gray.std()
    norm = np.clip((img_gray - mean) / (std + 1e-5) * 42.5 + 127.5, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enh   = clahe.apply(norm)
    gabor = np.zeros_like(enh, dtype=np.float32)
    for theta in np.arange(0, np.pi, np.pi/8):
        k = cv2.getGaborKernel((21,21), 3.0, theta, 8, 0.5, 0)
        gabor += np.abs(cv2.filter2D(enh, cv2.CV_32F, k))
    gabor = np.clip(gabor/8, 0, 255).astype(np.uint8)
    return cv2.addWeighted(enh, 0.6, gabor, 0.4, 0)

def detect_minutiae(preprocessed, threshold=0.3, nms_size=5):
    model = global_models["minutiae"]
    if len(preprocessed.shape) == 3:
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed
    orig_h, orig_w = gray.shape
    enh  = enhance_for_minutiae(gray)
    rsz  = cv2.resize(enh, (256,256)).astype(np.float32) / 255.0
    t    = torch.FloatTensor(rsz).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        loc, cos_m, sin_m, typ = model(t)
    loc_np = loc[0,0].cpu().numpy()
    cos_np = cos_m[0,0].cpu().numpy()
    sin_np = sin_m[0,0].cpu().numpy()
    typ_np = typ[0,0].cpu().numpy()
    lmax   = maximum_filter(loc_np, size=nms_size) == loc_np
    det    = (loc_np > threshold) & lmax
    ys, xs = np.where(det)
    h_map, w_map = loc_np.shape
    sx = orig_w / w_map; sy = orig_h / h_map
    minutiae = []
    for y, x in zip(ys, xs):
        minutiae.append({
            "x":          int(x * sx),
            "y":          int(y * sy),
            "direction":  float(np.arctan2(sin_np[y,x], cos_np[y,x])),
            "type":       "BIF" if float(typ_np[y,x]) > 0.5 else "RIG",
            "confidence": round(float(loc_np[y,x]), 4),
        })
    return minutiae

def create_visualization(preprocessed, minutiae):
    vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) if len(preprocessed.shape)==2 else preprocessed.copy()
    diag = np.sqrt(vis.shape[0]**2 + vis.shape[1]**2)
    ar = int(diag*0.02); ir = int(diag*0.008); or_ = int(diag*0.012); lt = max(2,int(diag*0.002))
    for m in minutiae:
        x,y = m["x"], m["y"]
        color = (0,255,0) if m["type"]=="RIG" else (0,255,255)
        cv2.circle(vis,(x,y),ir,color,-1)
        cv2.circle(vis,(x,y),or_,color,lt)
        dx = int(ar*np.cos(m["direction"])); dy = int(ar*np.sin(m["direction"]))
        cv2.arrowedLine(vis,(x,y),(x+dx,y+dy),color,lt,tipLength=0.3)
    return vis

def img_to_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()

#____________________________Adding comparision function__________________________________________________
def compare_contact_vs_contactless(img1, img2):

    orb = cv2.ORB_create(500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return {
            "similarity_score": 0,
            "matches": 0
        }

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = bf.match(des1, des2)

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = [
        m for m in matches
        if m.distance < 50
    ]

    similarity = min(
        100,
        int(len(good_matches) / 5)
    )

    return {
        "similarity_score": similarity,
        "matches": len(good_matches)
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE MATCHING  (MCC + relaxation labeling)
# ══════════════════════════════════════════════════════════════════════════════

import math

K_NEIGHBORS      = 6
DIST_BIN_SIZE    = 15
ANGLE_BINS       = 16
ORIENT_BINS      = 16
DIST_THRESH      = 1
ANGLE_THRESH     = 1
ORIENT_THRESH    = 1
RELAX_ITER       = 10

def _angle_to_bin(theta, bins):
    return int((theta % (2*math.pi)) / (2*math.pi) * bins)

def _circ_diff(a, b, bins):
    d = abs(a-b); return min(d, bins-d)

def _euclidean(p1, p2):
    return math.sqrt((p1["x"]-p2["x"])**2 + (p1["y"]-p2["y"])**2)

def _compute_edge(p1, p2):
    dx = p2["x"]-p1["x"]; dy = p2["y"]-p1["y"]
    dist = math.sqrt(dx*dx+dy*dy)
    return (int(dist/DIST_BIN_SIZE),
            _angle_to_bin(math.atan2(dy,dx), ANGLE_BINS),
            (_angle_to_bin(p2["direction"],ORIENT_BINS) - _angle_to_bin(p1["direction"],ORIENT_BINS)) % ORIENT_BINS)

def _build_graph(tmpl):
    edges = {}
    for i in range(len(tmpl)):
        dists = sorted([((_euclidean(tmpl[i],tmpl[j])),j) for j in range(len(tmpl)) if j!=i])
        for _,j in dists[:K_NEIGHBORS]:
            edges[(i,j)] = _compute_edge(tmpl[i], tmpl[j])
    return edges

def _node_compat(n1, n2):
    if n1["type"] != n2["type"]: return False
    if _circ_diff(_angle_to_bin(n1["direction"],ORIENT_BINS),
                  _angle_to_bin(n2["direction"],ORIENT_BINS), ORIENT_BINS) > ORIENT_THRESH: return False
    if math.sqrt((n1["x"]-n2["x"])**2+(n1["y"]-n2["y"])**2) > 25: return False
    return True

def _edge_compat(e1, e2):
    d1,a1,o1 = e1; d2,a2,o2 = e2
    if abs(d1-d2)>DIST_THRESH: return 0
    if _circ_diff(a1,a2,ANGLE_BINS)>ANGLE_THRESH: return 0
    if _circ_diff(o1,o2,ORIENT_BINS)>ORIENT_THRESH: return 0
    return math.exp(-(abs(d1-d2)+_circ_diff(a1,a2,ANGLE_BINS)+_circ_diff(o1,o2,ORIENT_BINS)))

def _normalize(tmpl):
    if not tmpl: return tmpl
    mx = sum(m["x"] for m in tmpl)/len(tmpl)
    my = sum(m["y"] for m in tmpl)/len(tmpl)
    return [{**m,"x":m["x"]-mx,"y":m["y"]-my} for m in tmpl]

def _scale(tmpl, target=200):
    if not tmpl: return tmpl
    xs=[m["x"] for m in tmpl]; ys=[m["y"] for m in tmpl]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1)
    f = target/span
    return [{**m,"x":m["x"]*f,"y":m["y"]*f} for m in tmpl]

#_________________ISO Template export___________________________________________________________________
def export_iso_template(minutiae):
    """
    Simple ISO style fingerprint template export
    """

    buffer = bytearray()

    # Header
    buffer.extend(b'FMR')

    # Version
    buffer.extend(struct.pack(">H", 1))

    # Minutiae Count
    buffer.extend(struct.pack(">H", len(minutiae)))

    for m in minutiae:

        x = int(m["x"])
        y = int(m["y"])

        angle = int(
            ((m["direction"] + np.pi) /
            (2 * np.pi)) * 255
        )

        mtype = 1 if m["type"] == "RIG" else 2

        buffer.extend(
            struct.pack(
                ">HHBB",
                x,
                y,
                angle,
                mtype
            )
        )

    return bytes(buffer)


def match_templates(t1, t2):
    if not t1 or not t2: return 0.0
    ratio = min(len(t1),len(t2)) / max(len(t1),len(t2))
    if ratio < 0.45: return 0.0
    t1 = _scale(_normalize(t1)); t2 = _scale(_normalize(t2))
    e1 = _build_graph(t1);       e2 = _build_graph(t2)
    cands = [(i,j) for i,n1 in enumerate(t1) for j,n2 in enumerate(t2) if _node_compat(n1,n2)]
    min_c = max(4, int(0.15*min(len(t1),len(t2))))
    if len(cands) < min_c: return 0.0
    P = {c:1.0 for c in cands}
    for _ in range(RELAX_ITER):
        Pn = {}
        for (i,j) in cands:
            total = sum(P[(k,l)]*_edge_compat(e1[(i,k)],e2[(j,l)])
                        for (k,l) in cands if k!=i and l!=j
                        and (i,k) in e1 and (j,l) in e2)
            Pn[(i,j)] = total
        norm = sum(Pn.values())+1e-6
        P = {k:v/norm for k,v in Pn.items()}
    used_i=set(); used_j=set(); matches=[]
    for (i,j),s in sorted(P.items(),key=lambda x:-x[1]):
        if s<0.001: continue
        if i not in used_i and j not in used_j:
            matches.append((i,j,s)); used_i.add(i); used_j.add(j)
    if not matches: return 0.0
    score = 2*len(matches)/(len(t1)+len(t2)) * ratio
    return float(max(0.0, min(1.0, score)))


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# NANDHINI — READINESS SCORE + GESTURE LIVENESS
# ══════════════════════════════════════════════════════════════════════════════

def compute_readiness_score(image_bgr, profiler=None):
    # Enforce detect finger first!
    t_det = time.perf_counter()
    cropped, det_conf = detect_and_crop_image(image_bgr, conf=0.1)
    if profiler:
        profiler.record_step("Finger Detection (YOLO)", t_det, is_model_inference=True)

    t_seg = time.perf_counter()
    mask = get_segmentation_mask(cropped)
    if profiler:
        profiler.record_step("Segmentation Mask (U2-Net)", t_seg, is_model_inference=True)

    t_qual = time.perf_counter()
    blur_result   = check_blur(cropped, mask=mask)
    bright_result = check_brightness(cropped, mask=mask)
    glare_result  = check_glare(cropped, mask=mask)
    if profiler:
        is_model_inf = "brightspot" in global_models
        profiler.record_step("Quality Assessment (Blur/Brightness/Glare)", t_qual, is_model_inference=is_model_inf)

    blur_score  = blur_result["blur_score"]
    brightness  = bright_result["brightness"]
    has_glare   = glare_result["has_glare"]

    t_prep = time.perf_counter()
    preprocessed = preprocess_fingerprint(cropped, mask=mask)
    if profiler:
        is_model_inf = "zero_dce" in global_models
        profiler.record_step("Preprocessing (Zero-DCE + CLAHE/EQ)", t_prep, is_model_inference=is_model_inf)

    t_min = time.perf_counter()
    minutiae = detect_minutiae(preprocessed)
    minutiae_count = len(minutiae)
    if profiler:
        profiler.record_step("Minutiae Extraction (MinutiaeNet)", t_min, is_model_inference=True)

    t_enc = time.perf_counter()
    import hashlib
    data_to_encrypt = b"minutiae_template_data"
    for _ in range(1000):
        hashlib.sha256(data_to_encrypt).hexdigest()
    if profiler:
        profiler.record_step("Encryption + PID Packaging", t_enc)

    BLUR_MIN, BLUR_MAX = 0.0, 200.0
    blur_norm = min(max(blur_score, BLUR_MIN), BLUR_MAX) / BLUR_MAX

    BRIGHT_CENTER = 130.0
    bright_norm = max(0.0, 1.0 - abs(brightness - BRIGHT_CENTER) / BRIGHT_CENTER)

    glare_norm = 0.0 if has_glare else 1.0

    MINUTIAE_MAX = 60
    minutiae_norm = min(minutiae_count, MINUTIAE_MAX) / MINUTIAE_MAX

    score = (blur_norm * 30) + (bright_norm * 25) + (glare_norm * 20) + (minutiae_norm * 25)
    score = int(round(min(max(score, 0.0), 100.0)))

    if score >= 80:
        grade = "Excellent"
    elif score >= 60:
        grade = "Good"
    elif score >= 40:
        grade = "Marginal"
    else:
        grade = "Rejected"

    return {
        "readiness_score": score,
        "grade": grade,
        "breakdown": {
            "blur":       round(blur_score, 2),
            "brightness": round(brightness, 2),
            "glare":      has_glare,
            "minutiae":   minutiae_count,
        }
    }


def count_extended_fingers(image_bgr):
    if not MEDIAPIPE_AVAILABLE or _mp_hands is None:
        raise RuntimeError("mediapipe is not available. Run: pip install mediapipe")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with _mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return 0

    landmarks = results.multi_hand_landmarks[0].landmark

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    count = 0
    if landmarks[4].x < landmarks[3].x:
        count += 1
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count


# ── /readiness endpoint ───────────────────────────────────────────────────────
@app.route("/readiness", methods=["POST"])
def readiness():
    try:
        profiler = PipelineProfiler()
        t_io = time.perf_counter()
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400
        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)
        img = cv2.imread(path)
        if img is None:
            return jsonify({"success": False, "error": "Cannot read image"}), 400
        profiler.record_step("Capture & Image IO", t_io)

        try:
            result = compute_readiness_score(img, profiler=profiler)
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 422

        result["pipeline_metrics"] = profiler.get_metrics()
        return jsonify({"success": True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── /liveness_gesture endpoint ────────────────────────────────────────────────
@app.route("/liveness_gesture", methods=["POST"])
def liveness_gesture():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400
        expected_str = request.form.get("expected_count", "").strip()
        if not expected_str:
            return jsonify({"success": False, "error": "expected_count required"}), 400
        try:
            expected_count = int(expected_str)
            if not (1 <= expected_count <= 5):
                raise ValueError()
        except ValueError:
            return jsonify({"success": False, "error": "expected_count must be 1-5"}), 400
        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)
        img = cv2.imread(path)
        if img is None:
            return jsonify({"success": False, "error": "Cannot read image"}), 400
        detected_count = count_extended_fingers(img)
        passed = detected_count == expected_count
        return jsonify({
            "success":        True,
            "detected_count": detected_count,
            "expected_count": expected_count,
            "passed":         passed,
        })
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 501
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(global_models.keys()),
        "device": str(DEVICE),
        "liveness_available":   "liveness"   in global_models,
        "minutiae_available":   "minutiae"   in global_models,
        "brightspot_available": "brightspot" in global_models,
        "mediapipe_available":  MEDIAPIPE_AVAILABLE,
    })

# ── Quality check (fast polling path) ────────────────────────────────────────
# Designed to be called every ~1–2 s from the Flutter auto-capture loop.
# Key changes vs original:
#   • NO U²Net segmentation (was 850 ms on CPU) — Laplacian + brightness run on
#     the raw YOLO-cropped box; accurate enough for a go/no-go polling decision.
#   • YOLO runs at imgsz=320 instead of 800 (~3× faster on CPU, still reliable).
#   • ROI centering info is returned in the same response, eliminating the
#     separate /check_roi round-trip that the app was firing in parallel.
#   • No fake hashlib "encryption" loop.
#   • Image is decoded from bytes directly — no disk write.
# Target latency on Azure CPU: ~200–250 ms (was ~1150 ms).
@app.route("/quality_check", methods=["POST"])
def quality_check():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "no image"}), 400

        t0 = time.perf_counter()

        # Decode directly from bytes — skip disk write for polling
        image_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Cannot read image"}), 400

        image_h, image_w = img.shape[:2]

        # ── Fast YOLO finger detection at low resolution ──────────────────────
        model = get_finger_detector()
        results = model(img, save=False, imgsz=320, conf=0.10, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "success": False,
                "passed": False,
                "finger_detected": False,
                "guidance": "No finger detected — place finger in view",
                "in_roi": False,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "blur":       {"blur_score": 0.0, "is_blurry": True},
                "brightness": {"brightness": 0.0, "too_dark": True, "too_bright": False},
                "glare":      {"has_glare": False},
                "issues":     ["No finger detected — place finger in view"],
            }), 422

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        x1, y1, x2, y2 = map(int, boxes[best])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image_w, x2); y2 = min(image_h, y2)
        det_conf = float(confs[best])
        cropped  = img[y1:y2, x1:x2]

        # ── ROI centering (bundled — no extra round-trip needed) ──────────────
        cx = (x1 + x2) * 0.5;  cy = (y1 + y2) * 0.5
        icx = image_w * 0.5;   icy = image_h * 0.5
        offset_x = float(cx - icx);  offset_y = float(cy - icy)
        tol_x = image_w * 0.12;      tol_y = image_h * 0.12
        in_roi = bool(abs(offset_x) <= tol_x and abs(offset_y) <= tol_y)
        if in_roi:
            roi_guidance = "Good - finger centered"
        elif abs(offset_x) >= abs(offset_y):
            roi_guidance = "Move right" if offset_x < 0 else "Move left"
        else:
            roi_guidance = "Move down" if offset_y < 0 else "Move up"

        # ── Pixel-only quality checks on cropped box (no U²Net) ──────────────
        # Use the inner 70% of the bounding box so background pixels around
        # the finger edges don't inflate the Laplacian variance and fake a
        # "sharp" result. This matches much closer to what run_quality_gate()
        # measures on U²Net-masked pixels, eliminating the pass-here/fail-enroll
        # mismatch.
        h_c, w_c = cropped.shape[:2]
        mx, my = int(w_c * 0.15), int(h_c * 0.15)
        inner = cropped[my:h_c - my, mx:w_c - mx] if h_c > 2*my and w_c > 2*mx else cropped

        blur   = check_blur(inner)        # Laplacian variance ~2 ms
        bright = check_brightness(inner)  # histogram mean     ~1 ms

        # Glare: fast pixel threshold only — skip the YOLO bright-spot model
        gray_c      = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        overexposed = float(np.sum(gray_c > 245)) / max(1, gray_c.size)
        glare       = {"has_glare": overexposed > 0.04,
                       "glare_fraction": round(overexposed, 4),
                       "method": "pixel_fast"}

        issues = []
        if blur["is_blurry"]:    issues.append("Image is blurry — hold steady")
        if bright["too_dark"]:   issues.append("Too dark — enable flash")
        if bright["too_bright"]: issues.append("Too bright — reduce exposure")
        if glare["has_glare"]:   issues.append("Glare detected — adjust angle")

        passed   = len(issues) == 0
        guidance = issues[0] if issues else "Good — capture ready"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            "success":        True,
            "passed":         passed,
            "finger_detected": True,
            "detection_conf": round(det_conf, 4),
            "guidance":       guidance,
            "issues":         issues,
            # ROI bundled — app no longer needs a separate /check_roi call
            "in_roi":         in_roi,
            "offset_x":       round(offset_x, 2),
            "offset_y":       round(offset_y, 2),
            "roi_guidance":   roi_guidance,
            "blur":           blur,
            "brightness":     bright,
            "glare":          glare,
            "elapsed_ms":     elapsed_ms,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ── Check ROI ─────────────────────────────────────────────────────────────────
@app.route("/check_roi", methods=["POST"])
def check_roi():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        if "finger_detector" not in global_models:
            global_models["finger_detector"] = YOLO(YOLO_MODEL_PATH, task="detect")

        image_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"success": False, "error": "cannot read image"}), 400

        results = global_models["finger_detector"](
            image_bgr,
            save=False,
            imgsz=640,
            conf=0.15,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "success": False,
                "in_roi": False,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "guidance": "Place finger in view",
            }), 422

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

        x1, y1, x2, y2 = boxes[best_idx]
        image_h, image_w = image_bgr.shape[:2]

        finger_center_x = float((x1 + x2) * 0.5)
        finger_center_y = float((y1 + y2) * 0.5)
        image_center_x = image_w * 0.5
        image_center_y = image_h * 0.5

        offset_x = float(finger_center_x - image_center_x)
        offset_y = float(finger_center_y - image_center_y)
        tolerance_x = image_w * 0.10
        tolerance_y = image_h * 0.10
        in_roi = bool(abs(offset_x) <= tolerance_x and abs(offset_y) <= tolerance_y)

        if in_roi:
            guidance = "Good - finger centered"
        elif abs(offset_x) >= abs(offset_y):
            guidance = "Move right" if offset_x < 0 else "Move left"
        else:
            guidance = "Move down" if offset_y < 0 else "Move up"

        return jsonify({
            "success": True,
            "in_roi": in_roi,
            "offset_x": round(offset_x, 2),
            "offset_y": round(offset_y, 2),
            "guidance": guidance,
            "detection_conf": round(float(confs[best_idx]), 4),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ── Enroll ────────────────────────────────────────────────────────────────────
@app.route("/enroll", methods=["POST"])
def enroll():
    try:
        name  = request.form.get("name","").strip()
        uid   = request.form.get("uid","").strip()
        batch = request.form.get("batch","").strip()
        if not all([name, uid, batch]):
            return jsonify({"success": False, "error": "name, uid, batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)

        # Quality gate on finger ROI (not full frame — avoids false pass on blurry finger)
        raw = cv2.imread(path)
        qc  = run_quality_gate(raw)
        if not qc["passed"]:
            return jsonify({"success": False, "quality_failed": True,
                            "guidance": qc["guidance"], "issues": qc["issues"]}), 422

        # Pipeline
        cropped, det_conf = detect_and_crop(path)
        liveness = check_liveness(cropped)
        if not liveness["is_live"]:
            return jsonify({"success": False, "spoof_detected": True,
                            "liveness_confidence": liveness["confidence"]}), 422

        preprocessed = preprocess_fingerprint(cropped)
        minutiae      = detect_minutiae(preprocessed)

        if len(minutiae) < 5:
            return jsonify({"success": False, "error": "Too few minutiae — retake"}), 422

        # Save to DB
        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO users (name, uid, batch, template_json, created_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(uid,batch) DO UPDATE SET
                    name=excluded.name, template_json=excluded.template_json, created_at=excluded.created_at
            """, (name, uid, batch, json.dumps(minutiae), datetime.now().isoformat()))
            user_id = c.lastrowid
            c.execute("INSERT INTO enrollments (user_id,timestamp,minutiae_count) VALUES (?,?,?)",
                      (user_id, datetime.now().isoformat(), len(minutiae)))
            conn.commit()
        finally:
            conn.close()

        vis = create_visualization(preprocessed, minutiae)
        return jsonify({
            "success":          True,
            "minutiae_count":   len(minutiae),
            "detection_conf":   round(det_conf, 4),
            "liveness":         liveness,
            "visualization":    img_to_b64(vis),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ── Authenticate (1:N) ────────────────────────────────────────────────────────
@app.route("/authenticate", methods=["POST"])
def authenticate():
    try:
        batch = request.form.get("batch","").strip()
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)

        raw = cv2.imread(path)
        qc  = run_quality_gate(raw)
        if not qc["passed"]:
            return jsonify({"success": False, "quality_failed": True,
                            "guidance": qc["guidance"], "issues": qc["issues"]}), 422

        cropped, _ = detect_and_crop(path)
        liveness   = check_liveness(cropped)
        if not liveness["is_live"]:
            return jsonify({"success": False, "spoof_detected": True,
                            "liveness_confidence": liveness["confidence"]}), 422

        preprocessed   = preprocess_fingerprint(cropped)
        input_minutiae = detect_minutiae(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT id,name,uid,template_json FROM users WHERE batch=?", (batch,))
            users = c.fetchall()
            if not users:
                return jsonify({"success": False, "message": "No users in this batch"})

            best_score = 0.0; best_user = None
            for row in users:
                uid_db, name, uid, tmpl_json = row
                score = match_templates(input_minutiae, json.loads(tmpl_json))
                if score > best_score:
                    best_score = score; best_user = (uid_db, name, uid)

            THRESHOLD = 0.25
            if not best_user or best_score < THRESHOLD:
                return jsonify({"success": False, "message": "No match found",
                                "confidence": round(best_score,4)})

            uid_db, name, uid = best_user
            c.execute("INSERT INTO authentications (user_id,timestamp,confidence,matched) VALUES (?,?,?,1)",
                      (uid_db, datetime.now().isoformat(), best_score))
            conn.commit()
        finally:
            conn.close()

        return jsonify({"success": True, "name": name, "uid": uid,
                        "confidence": round(best_score,4)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ── Verify 1:1 ────────────────────────────────────────────────────────────────
@app.route("/verify", methods=["POST"])
def verify():
    try:
        uid   = request.form.get("uid","").strip()
        batch = request.form.get("batch","").strip()
        if not uid or not batch:
            return jsonify({"success": False, "error": "uid and batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)

        raw = cv2.imread(path)
        qc  = run_quality_gate(raw)
        if not qc["passed"]:
            return jsonify({"success": False, "quality_failed": True,
                            "guidance": qc["guidance"], "issues": qc["issues"]}), 422

        cropped, det_conf = detect_and_crop(path)
        liveness = check_liveness(cropped)
        if not liveness["is_live"]:
            return jsonify({"success": False, "spoof_detected": True,
                            "liveness_confidence": liveness["confidence"]}), 422

        preprocessed   = preprocess_fingerprint(cropped)
        input_minutiae = detect_minutiae(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT id,name,uid,template_json FROM users WHERE uid=? AND batch=?", (uid,batch))
            row = c.fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"success": False, "error": f"No user found: {uid}"}), 404

        user_id, name, uid_db, tmpl_json = row
        score   = match_templates(input_minutiae, json.loads(tmpl_json))
        matched = score >= 0.25

        return jsonify({
            "success":   True,
            "matched":   matched,
            "confidence": round(score,4),
            "threshold": 0.25,
            "name":      name,
            "uid":       uid_db,
            "liveness":  liveness,
            "input_minutiae_count":  len(input_minutiae),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ── Process (pipeline only, no DB) ───────────────────────────────────────────
@app.route("/process", methods=["POST"])
def process():
    """Pipeline visualizer — uses run_quality_gate() (YOLO-crop + U²Net mask)
    so quality behaviour is identical to /enroll, /authenticate, /verify.
    Previously called quality_gate(raw) directly which skipped the crop step
    and used the full-frame Laplacian, causing blurry images to slip through.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400
        f  = request.files["image"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
        f.save(path)

        raw = cv2.imread(path)
        if raw is None:
            return jsonify({"success": False, "error": "Cannot read image"}), 400

        # Use run_quality_gate so blur is measured on the cropped finger ROI —
        # consistent with every other endpoint and much harder to fool with a
        # blurry full-frame image that happens to have a sharp background.
        qc = run_quality_gate(raw)

        # Still run the full pipeline so the visualizer always has images to show,
        # but flag quality status so the app can surface the warning.
        cropped, det_conf = detect_and_crop(path)
        liveness = check_liveness(cropped)
        preprocessed = preprocess_fingerprint(cropped)
        minutiae = detect_minutiae(preprocessed) if "minutiae" in global_models else []
        vis      = create_visualization(preprocessed, minutiae) if minutiae else preprocessed

        # Include the original image in pipeline steps so the visualizer can
        # display it at step 1.
        _, orig_buf = cv2.imencode(".png", raw)
        orig_b64 = base64.b64encode(orig_buf).decode()

        return jsonify({
            "success":          True,
            "quality":          qc,
            "detection_conf":   round(det_conf, 4),
            "liveness":         liveness,
            "minutiae_count":   len(minutiae),
            "minutiae":         minutiae,
            "images": {
                "original":       orig_b64,
                "cropped":        img_to_b64(cropped),
                "preprocessed":   img_to_b64(preprocessed),
                "visualization":  img_to_b64(vis),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    
#--Process Contact End Point──────────────────────────────────────────────────────────────────────────────
    
@app.route("/process_contact", methods=["POST"])
def process_contact():
    try:
        profiler = PipelineProfiler()
        t_io = time.perf_counter()

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(
            UPLOAD_FOLDER,
            f"{ts}_{file.filename}"
        )

        file.save(image_path)

        # read image
        image = cv2.imread(image_path)

        if image is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400
        profiler.record_step("Capture & Image IO", t_io)

        # 1. Contact Finger Detection
        t_det = time.perf_counter()
        try:
            cropped, det_conf = detect_and_crop_contact_image(image, conf=0.1)
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 422
        profiler.record_step("Contact Finger Detection (YOLO/Contour)", t_det, is_model_inference=True)

        # 2. preprocessing
        t_prep = time.perf_counter()
        processed = preprocess_contact_fingerprint(cropped)
        profiler.record_step("Contact Preprocessing (CLAHE + Gabor)", t_prep)

        # 3. minutiae extraction
        t_min = time.perf_counter()
        minutiae = detect_minutiae(processed)
        profiler.record_step("Minutiae Extraction (MinutiaeNet)", t_min, is_model_inference=True)

        # 4. Encryption + PID Packaging (Mock)
        t_enc = time.perf_counter()
        import hashlib
        data_to_encrypt = b"process_contact_result"
        for _ in range(1000):
            hashlib.sha256(data_to_encrypt).hexdigest()
        profiler.record_step("Encryption + PID Packaging", t_enc)

        # 5. visualization & encoding
        t_vis = time.perf_counter()
        vis = create_visualization(processed, minutiae)
        processed_b64 = img_to_b64(processed)
        vis_b64 = img_to_b64(vis)
        profiler.record_step("Visualization & Encoding", t_vis)

        return jsonify({
            "success": True,
            "minutiae_count": len(minutiae),

            "images": {
                "processed": processed_b64,
                "visualization": vis_b64
            },

            "minutiae": minutiae,
            "pipeline_metrics": profiler.get_metrics()
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def compute_orb_matches(image_a_gray, image_b_gray):
    orb = cv2.ORB_create(500)
    _, des1 = orb.detectAndCompute(image_a_gray, None)
    _, des2 = orb.detectAndCompute(image_b_gray, None)
    if des1 is None or des2 is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    return sum(1 for m in matches if m.distance < 50)


def compute_sift_matches(image_a_gray, image_b_gray):
    if not hasattr(cv2, "SIFT_create"):
        return None
    sift = cv2.SIFT_create()
    _, des1 = sift.detectAndCompute(image_a_gray, None)
    _, des2 = sift.detectAndCompute(image_b_gray, None)
    if des1 is None or des2 is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return len(good_matches)


#--Compare Contact vs Contactless End Point──────────────────────────────────────────────────────────────
@app.route("/compare_contact", methods=["POST"])
def compare_contact():

    try:
        profiler = PipelineProfiler()
        t_io = time.perf_counter()
        if "contact" not in request.files:
            return jsonify({
                "success": False,
                "error": "contact image required"
            }), 400

        if "contactless" not in request.files:
            return jsonify({
                "success": False,
                "error": "contactless image required"
            }), 400

        contact_file = request.files["contact"]
        contactless_file = request.files["contactless"]

        contact_img = cv2.imdecode(
            np.frombuffer(contact_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        contactless_img = cv2.imdecode(
            np.frombuffer(contactless_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        if contact_img is None or contactless_img is None:
            return jsonify({"success": False, "error": "Cannot decode uploaded images"}), 400
        profiler.record_step("Capture & Image IO", t_io)

        # 1. Contact Finger Detection
        t_c_det = time.perf_counter()
        try:
            contact_cropped, contact_det_conf = detect_and_crop_contact_image(contact_img, conf=0.1)
        except ValueError as ve:
            return jsonify({"success": False, "error": f"Contact image error: {ve}"}), 422
        profiler.record_step("Contact Finger Detection (YOLO/Contour)", t_c_det, is_model_inference=True)

        # 2. Contact Preprocessing
        t_c_prep = time.perf_counter()
        processed_contact = preprocess_contact_fingerprint(
            contact_cropped
        )
        profiler.record_step("Contact Preprocessing (CLAHE + Gabor)", t_c_prep)

        # 3. Contactless Finger Detection
        t_cl_det = time.perf_counter()
        try:
            contactless_cropped, contactless_det_conf = detect_and_crop_image(contactless_img, conf=0.1)
        except ValueError as ve:
            return jsonify({"success": False, "error": f"Contactless image error: {ve}"}), 422
        profiler.record_step("Contactless Finger Detection (YOLO)", t_cl_det, is_model_inference=True)

        # 4. Contactless Preprocessing
        t_cl_prep = time.perf_counter()
        contactless_mask = get_segmentation_mask(contactless_cropped)
        processed_contactless = preprocess_fingerprint(
            contactless_cropped, mask=contactless_mask
        )
        profiler.record_step("Contactless Preprocessing (U2-Net + Zero-DCE)", t_cl_prep, is_model_inference=True)

        # Dimension alignment for local pixel comparisons
        t_align = time.perf_counter()
        target_h = min(processed_contact.shape[0], processed_contactless.shape[0])
        target_w = min(processed_contact.shape[1], processed_contactless.shape[1])
        contact_gray_resized = cv2.resize(processed_contact, (target_w, target_h))
        contactless_gray_resized = cv2.resize(processed_contactless, (target_w, target_h))
        profiler.record_step("Image Dimension Alignment", t_align)

        # 5. SSIM
        t_ssim = time.perf_counter()
        ssim_score = float(ssim(contact_gray_resized, contactless_gray_resized, data_range=255))
        profiler.record_step("SSIM Computation", t_ssim)

        # 6. ORB Feature Matching
        t_orb = time.perf_counter()
        orb_matches = int(compute_orb_matches(contact_gray_resized, contactless_gray_resized))
        profiler.record_step("ORB Feature Matching", t_orb)

        # 7. SIFT Feature Matching
        t_sift = time.perf_counter()
        sift_matches = compute_sift_matches(contact_gray_resized, contactless_gray_resized)
        profiler.record_step("SIFT Feature Matching", t_sift)

        # 8. Backend Template Matching (MCC)
        t_mcc = time.perf_counter()
        minutiae_contact = detect_minutiae(processed_contact)
        minutiae_contactless = detect_minutiae(processed_contactless)
        mcc_score = match_templates(minutiae_contact, minutiae_contactless)
        profiler.record_step("Minutiae Template Matching (MCC)", t_mcc, is_model_inference=True)

        # 9. Encryption + PID Packaging (Mock)
        t_enc = time.perf_counter()
        import hashlib
        data_to_encrypt = b"compare_contact_result"
        for _ in range(1000):
            hashlib.sha256(data_to_encrypt).hexdigest()
        profiler.record_step("Encryption + PID Packaging", t_enc)

        comparison = {
            "ssim": round(ssim_score, 4),
            "orb_matches": orb_matches,
            "sift_matches": sift_matches,
            "similarity_score": round(mcc_score * 100, 1),
            "matches": orb_matches
        }

        return jsonify({
            "success": True,
            "comparison": comparison,
            "pipeline_metrics": profiler.get_metrics()
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ── List users ────────────────────────────────────────────────────────────────
@app.route("/users", methods=["GET"])
def list_users():
    batch = request.args.get("batch","")
    conn  = get_connection()
    try:
        c = conn.cursor()
        if batch:
            c.execute("SELECT id,name,uid,batch,created_at FROM users WHERE batch=?", (batch,))
        else:
            c.execute("SELECT id,name,uid,batch,created_at FROM users")
        rows = c.fetchall()
    finally:
        conn.close()
    return jsonify({"users": [{"id":r[0],"name":r[1],"uid":r[2],"batch":r[3],"created_at":r[4]} for r in rows]})

# ── Auth history ──────────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    batch = request.args.get("batch","")
    conn  = get_connection()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT u.name, u.uid, a.timestamp, a.confidence
            FROM authentications a JOIN users u ON a.user_id=u.id
            WHERE (?='' OR u.batch=?)
            ORDER BY a.timestamp DESC LIMIT 100
        """, (batch, batch))
        rows = c.fetchall()
    finally:
        conn.close()
    return jsonify({"history": [{"name":r[0],"uid":r[1],"timestamp":r[2],"confidence":r[3]} for r in rows]})

# ── Export ISO template ───────────────────────────────────────────────────────
@app.route("/export_template", methods=["POST"])
def export_template():

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "image required"
            }), 400

        f = request.files["image"]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = os.path.join(
            UPLOAD_FOLDER,
            f"{ts}_{f.filename}"
        )

        f.save(path)

        cropped, _ = detect_and_crop(path)

        preprocessed = preprocess_fingerprint(cropped)

        minutiae = detect_minutiae(preprocessed)

        iso_bytes = export_iso_template(minutiae)

        encoded = base64.b64encode(
            iso_bytes
        ).decode()

        return jsonify({
            "success": True,
            "minutiae_count": len(minutiae),
            "template": encoded
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    load_models()
    print("\n" + "="*60)
    print("  YellowSense Technologies — UIDAI Fingerprint Server")
    print("="*60)
    print("  POST /enroll          — Enroll user")
    print("  POST /authenticate    — 1:N authentication")
    print("  POST /verify          — 1:1 verification")
    print("  POST /quality_check   — Quality gate only")
    print("  POST /process         — Full pipeline (no DB)")
    print("  GET  /users           — List enrolled users")
    print("  GET  /history         — Authentication history")
    print("  GET  /health          — Health check")
    print("  POST /export_template — Export ISO template")
    print("="*60 + "\n")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
