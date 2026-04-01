from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Image dimensions (original simulator output)
ORIGINAL_HEIGHT = 160
ORIGINAL_WIDTH = 320

# Crop settings (remove sky and car hood)
CROP_TOP = 60
CROP_BOTTOM = 25

# PilotNet input size
PILOTNET_HEIGHT = 66
PILOTNET_WIDTH = 200

# ResNet input size
RESNET_SIZE = 224

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Camera steering offset for left/right cameras
CAMERA_STEERING_OFFSET = 0.2

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Steering loss weight (steering matters more than throttle)
STEERING_LOSS_WEIGHT = 5.0

# LSTM sequence length
SEQUENCE_LENGTH = 5

# Oversampling threshold for steering imbalance
STEERING_OVERSAMPLE_THRESHOLD = 0.1

# Orchestrator
CONFIDENCE_TEMPERATURE = 1.0
MIN_CONFIDENCE_THRESHOLD = 0.1
SAFE_THROTTLE = 0.2
