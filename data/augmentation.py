import cv2
import numpy as np


def random_flip(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    """Randomly flip image horizontally and negate steering angle."""
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def random_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly adjust brightness by converting to HSV and scaling V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 0.25 + np.random.random() * 0.75  # [0.25, 1.0]
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_shadow(image: np.ndarray) -> np.ndarray:
    """Add a random shadow across the image."""
    h, w = image.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.bool_)
    for row in range(h):
        boundary = int(x1 + (x2 - x1) * row / h)
        if np.random.random() > 0.5:
            mask[row, :boundary] = True
        else:
            mask[row, boundary:] = True
    hsv[mask, 2] *= 0.5
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_rotation(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Randomly rotate image within +/- max_angle degrees."""
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def augment(
    image: np.ndarray,
    steering: float,
    use_flip: bool = True,
    use_brightness: bool = True,
    use_shadow: bool = True,
    use_rotation: bool = False,
) -> tuple[np.ndarray, float]:
    """Apply a chain of random augmentations to the image."""
    if use_brightness:
        image = random_brightness(image)
    if use_shadow and np.random.random() > 0.5:
        image = random_shadow(image)
    if use_rotation:
        image = random_rotation(image)
    if use_flip:
        image, steering = random_flip(image, steering)
    return image, steering
