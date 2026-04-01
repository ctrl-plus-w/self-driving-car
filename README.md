# Multi-Agent Self-Driving Car — Steering & Throttle Prediction

A multi-agent system that predicts steering angle and throttle for a self-driving car using 4 distinct approaches, coordinated by a confidence-weighted orchestrator.

Built on the [Udacity Self-Driving Car Behavioral Cloning](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning/data) dataset (~20,000 simulator images from 3 cameras).

## The 4 Agents

| Agent | Architecture | Confidence Signal | Inference Speed |
|-------|-------------|-------------------|-----------------|
| **PilotNet** | NVIDIA end-to-end CNN (5 conv + 4 dense), YUV input | Variance of last 10 predictions | ~2ms |
| **ResNet18** | Pretrained ImageNet backbone + regression head, 2-phase training | Monte Carlo dropout (5 passes) | ~8ms |
| **Classical CV** | Canny/Hough lane detection + HSV histograms → XGBoost | Number of detected lane lines | <1ms |
| **CNN-LSTM** | CNN per-frame features + speed → LSTM over 5 frames | Hidden state norm stability | ~5ms |

## Orchestrator

The orchestrator collects `(steering, throttle, confidence)` from each agent, then:

1. Dampens confidence for agents with poor recent performance
2. Normalizes confidences via softmax with temperature
3. Computes a weighted average of predictions
4. Applies safety clamping (steering in [-1, 1], throttle in [0, 1])
5. Falls back to safe defaults if all confidences are below threshold

Agents that fail at inference (e.g. untrained) are silently skipped.

## Project Structure

```
tp-02/
  main.py                        # CLI entry point: train / evaluate / predict
  config/settings.py             # Hyperparameters & constants

  data/
    preprocessing.py             # Crop, resize, normalize (PilotNet & ResNet pipelines)
    augmentation.py              # Flip, brightness, shadow, rotation
    dataset.py                   # PyTorch Dataset with side-camera augmentation
    sequence_dataset.py          # Sliding window dataset for the LSTM agent

  agents/
    base_agent.py                # BaseAgent ABC + Prediction dataclass
    pilotnet_agent.py            # Agent 1: PilotNet
    resnet_agent.py              # Agent 2: ResNet18 transfer learning
    classical_agent.py           # Agent 3: Classical CV + XGBoost
    temporal_agent.py            # Agent 4: CNN-LSTM

  models/
    pilotnet.py                  # PilotNet nn.Module
    resnet_head.py               # ResNet18 + regression head
    feature_extractor.py         # Classical CV feature extraction
    cnn_lstm.py                  # CNN + LSTM nn.Module

  orchestrator/
    orchestrator.py              # Confidence-weighted ensemble
    confidence.py                # Softmax normalization + performance dampening
    safety.py                    # Clamping + fallback logic

  training/
    trainer.py                   # Generic PyTorch training loop
    train_pilotnet.py            # Train PilotNet
    train_resnet.py              # Train ResNet (2-phase)
    train_classical.py           # Train XGBoost
    train_temporal.py            # Train CNN-LSTM

  evaluation/
    metrics.py                   # MSE, MAE, steering smoothness
    evaluate.py                  # Per-agent + orchestrator evaluation
    visualize.py                 # Prediction plots + metric comparison charts
```

## Setup

```bash
pip install -e .
```

## Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning/data) and extract. The CSV contains columns: `centercam`, `leftcam`, `rightcam`, `steering_angle`, `throttle`, `reverse`, `speed`.

## Usage

### Train all agents

```bash
python main.py train --csv path/to/driving_log.csv --images path/to/IMG/
```

### Train a specific agent

```bash
python main.py train --agent pilotnet --csv path/to/driving_log.csv --images path/to/IMG/
```

Agent names: `pilotnet`, `resnet`, `classical_xgboost`, `temporal`

### Evaluate all agents + orchestrator

```bash
python main.py evaluate --csv path/to/driving_log.csv --images path/to/IMG/ --plot
```

### Single image prediction

```bash
python main.py predict --image path/to/frame.jpg --speed 15.0
```

## Training Details

- **Data split**: 70/15/15 (train/val/test) by sequential blocks to avoid temporal leakage
- **Steering imbalance**: Frames with |steering| > 0.1 are oversampled
- **Side cameras**: Left/right camera images used with +/-0.2 steering offset (triples training data)
- **Loss**: Weighted MSE — steering has 5x weight vs throttle
- **ResNet 2-phase**: Phase 1 trains head only (backbone frozen), Phase 2 fine-tunes last 2 blocks at lr/10
