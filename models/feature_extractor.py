"""Classical computer vision feature extraction for self-driving car images.

Extracts hand-crafted features (lane geometry, color histograms) from BGR
camera frames, suitable for feeding into a gradient-boosted tree model.
"""

import cv2
import numpy as np

from data.preprocessing import crop


# ---------------------------------------------------------------------------
# Lane features (Canny + Hough)
# ---------------------------------------------------------------------------

def extract_lane_features(image: np.ndarray) -> dict:
    """Detect lane lines via Canny edges + Hough transform and return geometric features.

    Parameters
    ----------
    image : np.ndarray
        Cropped BGR image (sky/hood already removed).

    Returns
    -------
    dict with keys:
        lane_center_offset   – horizontal offset of detected lane center from image center (normalised to [-1, 1])
        left_lane_angle      – average angle (radians) of lines in the left half
        right_lane_angle     – average angle (radians) of lines in the right half
        num_lines_detected   – total number of Hough lines found
        lane_curvature_estimate – rough curvature proxy (difference of left/right angles)
    """
    h, w = image.shape[:2]
    mid_x = w // 2

    # Convert to grayscale and blur before edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region-of-interest: bottom 2/3 of the (already-cropped) image
    roi_top = h // 3
    mask = np.zeros_like(edges)
    mask[roi_top:, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=60,
    )

    left_angles: list[float] = []
    right_angles: list[float] = []
    left_xs: list[float] = []
    right_xs: list[float] = []

    num_lines = 0

    if lines is not None:
        num_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            cx = (x1 + x2) / 2.0

            if cx < mid_x:
                left_angles.append(angle)
                left_xs.append(cx)
            else:
                right_angles.append(angle)
                right_xs.append(cx)

    left_angle = float(np.mean(left_angles)) if left_angles else 0.0
    right_angle = float(np.mean(right_angles)) if right_angles else 0.0

    # Estimate lane centre from average x positions of left and right lines
    left_center = float(np.mean(left_xs)) if left_xs else 0.0
    right_center = float(np.mean(right_xs)) if right_xs else float(w)
    lane_center = (left_center + right_center) / 2.0
    lane_center_offset = (lane_center - mid_x) / mid_x  # normalise to [-1, 1]

    curvature_estimate = right_angle - left_angle

    return {
        "lane_center_offset": lane_center_offset,
        "left_lane_angle": left_angle,
        "right_lane_angle": right_angle,
        "num_lines_detected": num_lines,
        "lane_curvature_estimate": curvature_estimate,
    }


# ---------------------------------------------------------------------------
# Colour histogram features
# ---------------------------------------------------------------------------

def extract_color_features(image: np.ndarray) -> np.ndarray:
    """Compute binned HSV histogram features from a cropped BGR image.

    Returns a 1-D numpy array of ~30 values (10 bins per H, S, V channel).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features: list[np.ndarray] = []

    bins_per_channel = 10
    ranges = [(0, 180), (0, 256), (0, 256)]  # H, S, V

    for ch_idx, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([hsv], [ch_idx], None, [bins_per_channel], [lo, hi])
        hist = hist.flatten().astype(np.float32)
        # Normalise so the sum equals 1 (proportion in each bin)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.append(hist)

    return np.concatenate(features)  # shape (30,)


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

def extract_all_features(
    image: np.ndarray,
    speed: float,
) -> tuple[np.ndarray, float]:
    """Build a flat feature vector from a raw BGR camera frame.

    Parameters
    ----------
    image : np.ndarray
        Raw BGR image straight from cv2.imread (before cropping).
    speed : float
        Current vehicle speed.

    Returns
    -------
    features : np.ndarray
        1-D float32 array  (5 lane + 30 colour + 1 speed = 36 features).
    lane_confidence : float
        Value in [0, 1] indicating how many lane lines were detected.
        0 means none; 1 means >= 10 lines (saturated).
    """
    cropped = crop(image)

    lane = extract_lane_features(cropped)
    color = extract_color_features(cropped)

    lane_vec = np.array(
        [
            lane["lane_center_offset"],
            lane["left_lane_angle"],
            lane["right_lane_angle"],
            lane["num_lines_detected"],
            lane["lane_curvature_estimate"],
        ],
        dtype=np.float32,
    )

    speed_vec = np.array([speed], dtype=np.float32)

    features = np.concatenate([lane_vec, color, speed_vec])

    # Confidence: saturate at 10 lines
    lane_confidence = float(np.clip(lane["num_lines_detected"] / 10.0, 0.0, 1.0))

    return features, lane_confidence
