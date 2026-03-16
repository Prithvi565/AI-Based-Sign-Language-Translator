"""
Indian Sign Language (ISL) — Live Webcam with Hand Landmarks
=============================================================
Install:
    pip install torch torchvision opencv-python pillow mediapipe

Run:
    python isl_webcam.py
    python isl_webcam.py --model_path "C:/path/to/best_isl_model.pth"

Controls:
    SPACE     → add letter
    ENTER     → add space
    BACKSPACE → delete last letter
    C         → clear sentence
    T         → toggle finger traces
    Q / ESC   → quit
"""

import sys
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import cv2
except ImportError:
    print("\n❌  Run:  pip install opencv-python\n")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("\n❌  Run:  pip install mediapipe\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE SETUP (Updated for new Tasks API)
# ══════════════════════════════════════════════════════════════════════════════

# Import the new MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Custom colours per finger (BGR format)
FINGER_TIPS   = [4, 8, 12, 16, 20]
FINGER_COLORS = {
    "thumb":  (0,   220, 255),   # yellow
    "index":  (0,   255, 80),    # green
    "middle": (255, 160, 0),     # blue
    "ring":   (255, 0,   200),   # purple
    "pinky":  (0,   80,  255),   # red
}

# Which landmark indices belong to each finger
FINGER_MAP = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

def get_lm_color(idx):
    for fname, indices in FINGER_MAP.items():
        if idx in indices:
            return FINGER_COLORS[fname]
    return (220, 220, 220)  # wrist / palm


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default=r"isl_output\best_isl_model.pth")
    p.add_argument("--cam_id",    type=int, default=0)
    p.add_argument("--box_size",  type=int, default=300)
    p.add_argument("--smoothing", type=int, default=7)
    p.add_argument("--trace_len", type=int, default=40)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    path = Path(model_path)
    assert path.exists(), f"\n❌  Model not found: {path}\n"

    print(f"\n  Loading: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    class_names = ckpt["class_names"]
    backbone    = ckpt["backbone"]
    img_size    = ckpt["img_size"]
    num_classes = ckpt["num_classes"]
    val_acc     = ckpt.get("val_acc", 0)

    print(f"  Backbone : {backbone}  |  {num_classes} classes  |  Val acc : {val_acc*100:.1f}%")
    print(f"  Device   : {DEVICE}" +
          (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    if backbone == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.2), nn.Linear(512, num_classes),
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_f  = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.2), nn.Linear(512, num_classes),
        )
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names, img_size


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, pil_img, class_names, transform):
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    top5_p, top5_i = probs.topk(min(5, len(class_names)))
    return {
        "label":      class_names[probs.argmax().item()],
        "confidence": probs.max().item(),
        "top5": [{"label": class_names[i.item()], "conf": p.item()}
                 for p, i in zip(top5_p, top5_i)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRAW HAND LANDMARKS  (the fixed version)
# ══════════════════════════════════════════════════════════════════════════════

def draw_landmarks(frame, hand_landmarks, trace_history, show_traces):
    """
    Draw hand skeleton directly onto frame using absolute pixel coordinates.
    Updated for new MediaPipe Tasks API.
    """
    h, w = frame.shape[:2]

    # ── Convert all 21 landmarks to pixel coords ──────────────────────────
    pts = {}
    for i, lm in enumerate(hand_landmarks):
        px = int(lm.x * w)
        py = int(lm.y * h)
        # Clamp to frame bounds
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        pts[i] = (px, py)

    # ── Draw all connections (bones) ──────────────────────────────────────
    # Standard hand connections (21 landmarks)
    connections = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]

    for conn in connections:
        a, b = conn
        if a in pts and b in pts:
            color = get_lm_color(a)
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # ── Draw landmark dots ────────────────────────────────────────────────
    for i, pt in pts.items():
        color  = get_lm_color(i)
        radius = 9 if i in FINGER_TIPS else 5

        # Outer dark ring for visibility
        cv2.circle(frame, pt, radius + 2, (0, 0, 0),   -1, cv2.LINE_AA)
        # Coloured fill
        cv2.circle(frame, pt, radius,     color,        -1, cv2.LINE_AA)
        # White centre dot on fingertips
        if i in FINGER_TIPS:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # ── Update and draw finger traces (only fingertips) ───────────────────
    for tip_idx in FINGER_TIPS:
        if tip_idx not in trace_history:
            trace_history[tip_idx] = deque(maxlen=40)
        trace_history[tip_idx].append(pts[tip_idx])

    if show_traces:
        for tip_idx, history in trace_history.items():
            color    = get_lm_color(tip_idx)
            pts_list = list(history)
            n        = len(pts_list)
            for j in range(1, n):
                alpha = j / n                             # fade older points
                c     = tuple(int(ch * alpha) for ch in color)
                thick = max(1, int(alpha * 3))
                cv2.line(frame, pts_list[j-1], pts_list[j],
                         c, thick, cv2.LINE_AA)

    return pts


def draw_finger_status(frame, hand_landmarks, x, y):
    """Show which fingers are up (▲) or down (▼)."""
    # hand_landmarks is now a list directly, not an object with .landmark
    lms = hand_landmarks

    def up(tip, pip):   return lms[tip].y < lms[pip].y
    def thumb_up():     return lms[4].x  > lms[3].x    # mirrored frame

    checks = [
        ("T", thumb_up()),
        ("I", up(8,  6)),
        ("M", up(12, 10)),
        ("R", up(16, 14)),
        ("P", up(20, 18)),
    ]

    # Background
    _overlay(frame, x - 5, y - 22, x + 200, y + 10, (20, 20, 20), 0.7)
    cv2.putText(frame, "Fingers:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

    for k, (name, extended) in enumerate(checks):
        col = (0, 210, 70) if extended else (60, 60, 200)
        sym = "U" if extended else "D"
        cv2.putText(frame, f"{name}{sym}",
                    (x + 75 + k * 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


def draw_model_preview(frame, roi, x, y, size=120):
    """Show what the model actually receives — greyscale thumbnail."""
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    preview = cv2.resize(gray, (size, size))
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(preview, (0,0), (size-1,size-1), (80,80,80), 1)

    # Paste
    fy2 = min(frame.shape[0], y + size)
    fx2 = min(frame.shape[1], x + size)
    frame[y:fy2, x:fx2] = preview[:fy2-y, :fx2-x]

    # Label
    _overlay(frame, x, y - 22, x + size, y, (20,20,20), 0.8)
    cv2.putText(frame, "Model input:", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)

    # Brightness check
    mean = preview[:,:,0].mean()
    if   mean < 40:  warn, wc = "TOO DARK",   (50,50,220)
    elif mean > 215: warn, wc = "TOO BRIGHT",  (50,220,220)
    else:            warn, wc = f"OK ({mean:.0f})", (0,200,80)
    _overlay(frame, x, y+size, x+size, y+size+22, (20,20,20), 0.7)
    cv2.putText(frame, warn, (x+4, y+size+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, wc, 1, cv2.LINE_AA)


def draw_conf_graph(frame, conf_hist, x, y, gw=200, gh=55):
    """Mini confidence line graph."""
    _overlay(frame, x-5, y-22, x+gw+5, y+gh+5, (20,20,20), 0.7)
    cv2.putText(frame, "Confidence history:", (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)

    pts_list = list(conf_hist)
    if len(pts_list) < 2:
        return
    n = len(pts_list)
    for i in range(1, n):
        x1 = x + int((i-1)/(n-1) * gw)
        x2 = x + int(i    /(n-1) * gw)
        y1 = y + gh - int(pts_list[i-1] * gh)
        y2 = y + gh - int(pts_list[i]   * gh)
        c  = (0,210,70) if pts_list[i] >= 0.85 else \
             (0,165,255) if pts_list[i] >= 0.60 else (50,50,210)
        cv2.line(frame, (x1,y1), (x2,y2), c, 2, cv2.LINE_AA)

    # 85% line
    ty = y + gh - int(0.85 * gh)
    cv2.line(frame, (x, ty), (x+gw, ty), (80,80,80), 1)
    cv2.putText(frame, "85%", (x+gw+3, ty+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)


def _overlay(frame, x1, y1, x2, y2, color, alpha):
    """Semi-transparent filled rectangle."""
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    sub  = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color[::-1])
    frame[y1:y2, x1:x2] = cv2.addWeighted(sub, 1-alpha, rect, alpha, 0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_webcam(model, class_names, img_size, cam_id, box_size, smoothing, trace_len):
    transform     = build_transform(img_size)
    pred_history  = deque(maxlen=smoothing)
    conf_history  = deque(maxlen=60)
    trace_history = {tip: deque(maxlen=trace_len) for tip in FINGER_TIPS}
    sentence      = []
    fps_buf       = deque(maxlen=30)
    show_traces   = True

    # ── Setup HandLandmarker with new Tasks API ──────────────────────────
    import urllib.request
    import os
    model_path = "hand_landmarker.task"
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(model_path):
        print("  Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("  Model downloaded.")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hands = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌  Cannot open camera {cam_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    FONT  = cv2.FONT_HERSHEY_DUPLEX
    FONTS = cv2.FONT_HERSHEY_SIMPLEX

    print("\n" + "═"*52)
    print("  🎥  ISL Webcam — Hand Landmark Mode")
    print("═"*52)
    print("  SPACE  →  add letter     ENTER  →  space")
    print("  BKSP   →  delete         C      →  clear")
    print("  T      →  toggle traces  Q/ESC  →  quit")
    print("═"*52 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Camera feed lost.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        fps_buf.append(time.time())

        # ── Capture box ───────────────────────────────────────────────────
        cx, cy = fw // 2, fh // 2
        bx1 = cx - box_size // 2
        by1 = cy - box_size // 2
        bx2 = bx1 + box_size
        by2 = by1 + box_size

        roi = frame[by1:by2, bx1:bx2].copy()

        # ══════════════════════════════════════════════════════════════════
        # MEDIAPIPE  — Updated for new Tasks API
        # ══════════════════════════════════════════════════════════════════
        rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp == 0:
            timestamp = int(time.time() * 1000)  # fallback timestamp

        mp_result = hands.detect_for_video(rgb_image, timestamp)

        hand_detected  = len(mp_result.hand_landmarks) > 0
        hand_landmarks = mp_result.hand_landmarks[0] if hand_detected else None

        # ── Draw landmarks DIRECTLY on frame ─────────────────────────────
        if hand_landmarks is not None:
            draw_landmarks(frame, hand_landmarks,
                           trace_history, show_traces)

        # ══════════════════════════════════════════════════════════════════
        # MODEL PREDICTION on ROI
        # ══════════════════════════════════════════════════════════════════
        pil    = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        result = predict(model, pil, class_names, transform)

        pred_history.append(result["label"])
        smoothed = max(set(pred_history), key=pred_history.count)
        conf     = result["confidence"]
        conf_history.append(conf)

        if   conf >= 0.85: box_col, ctag = (0,220,50),  "HIGH"
        elif conf >= 0.60: box_col, ctag = (0,165,255), "MED"
        else:              box_col, ctag = (50,50,220), "LOW"

        # ── TOP BAR ───────────────────────────────────────────────────────
        _overlay(frame, 0, 0, fw, 48, (15,15,15), 0.8)
        cv2.putText(frame,
            "SPACE=add  ENTER=space  BKSP=del  C=clear  T=traces  Q=quit",
            (12, 32), FONTS, 0.52, (155,155,155), 1, cv2.LINE_AA)
        if len(fps_buf) > 1:
            fps = len(fps_buf)/(fps_buf[-1]-fps_buf[0]+1e-6)
            cv2.putText(frame, f"FPS {fps:.0f}",
                        (fw-85, 32), FONTS, 0.58, (100,100,100), 1)

        # ── CAPTURE BOX ───────────────────────────────────────────────────
        # Draw box border (thick, rounded look with corner lines)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), box_col, 3)

        # Corner accents
        cl = 25  # corner line length
        cv2.line(frame, (bx1,by1), (bx1+cl,by1), (255,255,255), 2)
        cv2.line(frame, (bx1,by1), (bx1,by1+cl), (255,255,255), 2)
        cv2.line(frame, (bx2,by1), (bx2-cl,by1), (255,255,255), 2)
        cv2.line(frame, (bx2,by1), (bx2,by1+cl), (255,255,255), 2)
        cv2.line(frame, (bx1,by2), (bx1+cl,by2), (255,255,255), 2)
        cv2.line(frame, (bx1,by2), (bx1,by2-cl), (255,255,255), 2)
        cv2.line(frame, (bx2,by2), (bx2-cl,by2), (255,255,255), 2)
        cv2.line(frame, (bx2,by2), (bx2,by2-cl), (255,255,255), 2)

        cv2.putText(frame, "PLACE HAND HERE",
                    (bx1+10, by1+26), FONTS, 0.65, box_col, 2, cv2.LINE_AA)

        # Hand detected indicator
        det_col = (0,210,70) if hand_detected else (50,50,200)
        det_txt = "HAND DETECTED" if hand_detected else "NO HAND FOUND"
        _overlay(frame, bx1, by2+4, bx2, by2+34, (15,15,15), 0.75)
        cv2.putText(frame, det_txt,
                    (bx1+10, by2+24), FONTS, 0.65, det_col, 2, cv2.LINE_AA)

        # Trace toggle indicator
        tr_col = (0,200,70) if show_traces else (100,100,100)
        _overlay(frame, bx1, by2+36, bx2, by2+58, (15,15,15), 0.65)
        cv2.putText(frame, f"Traces: {'ON (press T to off)' if show_traces else 'OFF (press T to on)'}",
                    (bx1+10, by2+52), FONTS, 0.5, tr_col, 1, cv2.LINE_AA)

        # ── PREDICTION PANEL — above box ──────────────────────────────────
        _overlay(frame, bx1, by1-92, bx2, by1-4, (15,15,15), 0.78)
        cv2.putText(frame, smoothed,
                    (bx1+18, by1-28), FONT, 2.4, box_col, 4, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {conf*100:.1f}%  [{ctag}]",
                    (bx1+12, by1-8), FONTS, 0.65, (200,200,200), 1, cv2.LINE_AA)

        # ── RIGHT PANEL — top 5 ───────────────────────────────────────────
        rx = bx2 + 18
        if rx + 230 < fw:
            _overlay(frame, rx-8, by1, rx+235, by1+205, (15,15,15), 0.78)
            cv2.putText(frame, "TOP 5:", (rx, by1+22),
                        FONTS, 0.65, (200,200,200), 1, cv2.LINE_AA)
            for i, r in enumerate(result["top5"]):
                by_   = by1 + 50 + i * 30
                blen  = int(r["conf"] * 185)
                bcol  = box_col if i == 0 else (70,70,70)
                cv2.rectangle(frame, (rx, by_-16), (rx+blen, by_+4), bcol, -1)
                cv2.putText(frame,
                    f"{r['label']}  {r['conf']*100:.1f}%",
                    (rx+4, by_), FONTS, 0.65,
                    (255,255,255) if i == 0 else (155,155,155), 1, cv2.LINE_AA)

            # Finger status
            if hand_landmarks:
                draw_finger_status(frame, hand_landmarks, rx, by1 + 215)

            # Confidence graph
            draw_conf_graph(frame, conf_history, rx, by1+265, gw=215, gh=65)

        # ── LEFT PANEL — model preview ────────────────────────────────────
        lx = bx1 - 145
        if lx >= 0:
            draw_model_preview(frame, roi, lx, by1, size=132)

        # ── BOTTOM SENTENCE BAR ───────────────────────────────────────────
        _overlay(frame, 0, fh-85, fw, fh, (15,15,15), 0.85)
        sent_str = "".join(sentence)
        display  = ("..."+sent_str[-52:]) if len(sent_str)>52 else sent_str
        cv2.putText(frame, "Sentence:", (15, fh-58),
                    FONTS, 0.60, (130,130,130), 1, cv2.LINE_AA)
        cv2.putText(frame, display, (15, fh-18),
                    FONT, 1.1, (0,255,180), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(sent_str)} chars",
                    (fw-115, fh-18), FONTS, 0.52, (90,90,90), 1)

        cv2.imshow("ISL Sign Language  |  Hand Landmarks  (Q=quit)", frame)

        # ── KEY HANDLING ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if   key in (ord('q'), 27): break
        elif key == 32:
            sentence.append(smoothed)
            print(f"  + '{smoothed}'  →  {''.join(sentence)}")
        elif key == 13:
            sentence.append(" ")
            print(f"  + ' '  →  {''.join(sentence)}")
        elif key == 8:
            if sentence:
                r = sentence.pop()
                print(f"  - '{r}'  →  {''.join(sentence)}")
        elif key == ord('c'):
            sentence.clear()
            print("  Cleared.")
        elif key == ord('t'):
            show_traces = not show_traces
            print(f"  Traces: {'ON' if show_traces else 'OFF'}")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Final sentence: '{''.join(sentence)}'\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = get_args()
    model, class_names, img_size = load_model(args.model_path)
    run_webcam(
        model       = model,
        class_names = class_names,
        img_size    = img_size,
        cam_id      = args.cam_id,
        box_size    = args.box_size,
        smoothing   = args.smoothing,
        trace_len   = args.trace_len,
    )