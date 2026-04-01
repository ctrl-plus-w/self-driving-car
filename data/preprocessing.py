import cv2
import numpy as np

from config.settings import (
    CROP_BOTTOM,
    CROP_TOP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PILOTNET_HEIGHT,
    PILOTNET_WIDTH,
    RESNET_SIZE,
)


def crop(image: np.ndarray) -> np.ndarray:
    """Crop sky (top) and car hood (bottom) from the image."""
    return image[CROP_TOP : image.shape[0] - CROP_BOTTOM, :, :]


def resize_pilotnet(image: np.ndarray) -> np.ndarray:
    """Resize to PilotNet input dimensions (66x200)."""
    return cv2.resize(image, (PILOTNET_WIDTH, PILOTNET_HEIGHT))


def resize_resnet(image: np.ndarray) -> np.ndarray:
    """Resize to ResNet input dimensions (224x224)."""
    return cv2.resize(image, (RESNET_SIZE, RESNET_SIZE))


def to_yuv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to YUV color space (as in NVIDIA PilotNet paper)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def normalize_01(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization. Expects image in [0, 1] range, RGB order."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (image - mean) / std


def preprocess_pilotnet(image: np.ndarray) -> np.ndarray:
    """Full PilotNet preprocessing pipeline: crop -> resize -> YUV -> normalize."""
    image = crop(image)
    image = resize_pilotnet(image)
    image = to_yuv(image)
    image = normalize_01(image)
    return image


def preprocess_resnet(image: np.ndarray) -> np.ndarray:
    """Full ResNet preprocessing pipeline: crop -> resize -> RGB -> normalize."""
    image = crop(image)
    image = resize_resnet(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_01(image)
    image = normalize_imagenet(image)
    return image
