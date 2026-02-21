"""
Visual features extracted from the first 1-2 seconds of a YouTube Short.

Standalone module — no cross-imports from other scripts/.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Haar cascade shipped with opencv-python-headless
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_FACE_CASCADE_PATH)

_VISUAL_KEYS = [
    "v_brightness", "v_contrast", "v_edge_density", "v_colorfulness",
    "v_face_present", "v_face_area_frac", "v_text_area_frac",
    "v_motion_magnitude", "v_scene_cut",
]


def _none_dict() -> Dict[str, Any]:
    """Return a dict with all visual keys set to None (failure fallback)."""
    return {k: None for k in _VISUAL_KEYS}


# ---------------------------------------------------------------------------
# Video download
# ---------------------------------------------------------------------------

def download_short_video(video_id: str, output_path: Path) -> None:
    """Download the first ~3 seconds of a Short at 144p using yt-dlp.

    Tries format codes 160 → 133 → worstvideo in order.
    """
    url = f"https://www.youtube.com/shorts/{video_id}"
    for fmt in ("160", "133", "worstvideo"):
        cmd = [
            "yt-dlp",
            "-f", fmt,
            "--download-sections", "*0-3",
            "-o", str(output_path),
            "--no-playlist",
            "--quiet",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and output_path.exists():
            return
    raise RuntimeError(f"Failed to download video for {video_id}")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract frames at t=0.0s and t=1.0s from a video file.

    Returns (frame0, frame1) as BGR numpy arrays, or None if extraction fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    try:
        # Frame at t=0.0s
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok0, frame0 = cap.read()
        if not ok0:
            frame0 = None

        # Frame at t=1.0s
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok1, frame1 = cap.read()
        if not ok1:
            frame1 = None

        return frame0, frame1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _brightness(frame: np.ndarray) -> float:
    """Mean luminance (0-255) of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _contrast(frame: np.ndarray) -> float:
    """Standard deviation of luminance (0-~128)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def _edge_density(frame: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def _colorfulness(frame: np.ndarray) -> float:
    """Hasler-Süsstrunk colorfulness metric."""
    B, G, R = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def _face_detection(frame: np.ndarray) -> Tuple[bool, float]:
    """Detect faces using Haar cascade.

    Returns (face_present, face_area_fraction).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(faces) == 0:
        return False, 0.0
    total_area = sum(w * h for (_, _, w, h) in faces)
    frame_area = frame.shape[0] * frame.shape[1]
    return True, float(total_area / frame_area)


def _text_area(frame: np.ndarray) -> float:
    """Fraction of frame area covered by text-like regions (MSER)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return 0.0
    total_pixels = sum(len(r) for r in regions)
    frame_area = frame.shape[0] * frame.shape[1]
    return min(float(total_pixels / frame_area), 1.0)


def _motion_magnitude(frame0: np.ndarray, frame1: np.ndarray) -> float:
    """Mean optical flow magnitude between two frames (Farneback)."""
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _scene_cut(frame0: np.ndarray, frame1: np.ndarray, threshold: float = 0.5) -> bool:
    """Detect scene cut via histogram correlation below threshold."""
    hist0 = cv2.calcHist([frame0], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist0, hist0)
    cv2.normalize(hist1, hist1)
    corr = cv2.compareHist(hist0, hist1, cv2.HISTCMP_CORREL)
    return corr < threshold


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_visual_v1(video_path: Path) -> Dict[str, Any]:
    """Compute 9 visual features from a short video clip.

    Returns a flat dict with v_ prefixed keys. All values are None on failure.
    """
    try:
        frame0, frame1 = extract_frames(video_path)
        if frame0 is None:
            return _none_dict()

        face_present, face_area_frac = _face_detection(frame0)

        result: Dict[str, Any] = {
            "v_brightness": _brightness(frame0),
            "v_contrast": _contrast(frame0),
            "v_edge_density": _edge_density(frame0),
            "v_colorfulness": _colorfulness(frame0),
            "v_face_present": int(face_present),
            "v_face_area_frac": face_area_frac,
            "v_text_area_frac": _text_area(frame0),
        }

        if frame1 is not None:
            result["v_motion_magnitude"] = _motion_magnitude(frame0, frame1)
            result["v_scene_cut"] = int(_scene_cut(frame0, frame1))
        else:
            result["v_motion_magnitude"] = None
            result["v_scene_cut"] = None

        return result
    except Exception:
        return _none_dict()
