# Image Classification with DenseNet121 — Transfer Learning on CIFAR-10

> Deep learning project demonstrating end-to-end model development: transfer learning, iterative architecture refinement, and systematic performance evaluation on a 10-class image classification task.

---

## Overview

This project implements and enhances a **DenseNet121** convolutional neural network for image classification on the **CIFAR-10** dataset (60,000 images, 10 classes). The work follows a full ML development cycle — from data preprocessing and baseline modelling through diagnostic evaluation, architectural enhancement, and quantitative analysis of results.

The project emphasises skills directly applicable to quantitative and data-driven model development: **iterative model refinement driven by empirical evidence**, **robust evaluation methodology**, and **disciplined handling of data pipelines**.

---

## Skills Demonstrated

| Area | What this project covers |
|---|---|
| **Deep Learning** | Transfer learning, fine-tuning pretrained CNNs, custom layer design |
| **Model Architecture** | DenseNet dense connectivity, skip connections, feature reuse |
| **Optimisation** | Learning rate scheduling, early stopping, dropout regularisation |
| **Feature Engineering** | Data augmentation pipeline, input normalisation, upsampling strategy |
| **Evaluation** | Accuracy, precision, recall, F1-score, confusion matrix analysis |
| **Python / Libraries** | TensorFlow, Keras, scikit-learn, NumPy, Matplotlib |
| **Software Practice** | Modular code, PEP 8, reproducible pipeline, model serialisation |

---

## Dataset

**CIFAR-10** — 60,000 colour images (32×32 px) across 10 classes:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

| Split | Samples |
|---|---|
| Train | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |

Loaded via `tensorflow_datasets`. Test set kept completely held-out — never used during training or hyperparameter tuning.

---

## Project Structure

```
├── notebook.ipynb          # Full experiment — data → baseline → enhanced → evaluation
├── README.md               # This file
```

---

## Methodology

### 1. Data Pipeline

- Raw CIFAR-10 images (uint8, 0–255) converted to float32
- **DenseNet-specific normalisation** applied via a registered custom Keras layer — prevents the serialisation failure that causes preprocessing to silently drop at inference time
- Images upsampled from 32×32 → 224×224 inside the model graph, after augmentation, so spatial transforms operate at native resolution
- Labels one-hot encoded for categorical cross-entropy loss

```python
@keras.saving.register_keras_serializable()
class DenseNetPreprocess(layers.Layer):
    """Serialization-safe DenseNet normalization — replaces Lambda layer."""
    def call(self, inputs):
        return preprocess_input(inputs)
```

> **Key lesson:** Using `layers.Lambda(preprocess_input)` produces correct training behaviour but silently fails to deserialise when the model is reloaded — causing test accuracy to collapse from ~90% to 8%. The registered custom layer guarantees identical preprocessing at train and inference time.

---

### 2. Baseline Model

Frozen DenseNet121 backbone (pretrained on ImageNet) with a lightweight classification head:

```
Input (32×32×3)
→ Data augmentation     (RandomFlip, RandomTranslation, RandomZoom)
→ Rescaling × 255       (undo dataset /255 before DenseNet normalisation)
→ DenseNetPreprocess    (ImageNet channel-wise mean/std)
→ Resizing (224×224)
→ DenseNet121 (frozen)  (global average pooling)
→ Dense(256, ReLU)
→ Dropout(0.3)
→ Dense(10, Softmax)
```

| Metric | Value |
|---|---|
| Test accuracy | 90.28% |
| Test loss | 0.2919 |
| Macro F1 | 0.90 |

---

### 3. Diagnostic Evaluation → Modifications

Results from the baseline were treated as a **diagnostic tool**, not just a result. Each modification was motivated by specific evidence:

| Observation | Diagnosis | Modification |
|---|---|---|
| Val accuracy plateaued ~90%, train still climbing | Frozen backbone ceiling — block 4 calibrated for sharp ImageNet images, not blurry upsampled CIFAR-10 | Unfreeze `denseblock4` + `training=True` |
| Cat: 217 errors; dog/cat confusion: 153 errors | Single Dense(256) lacks capacity for visually similar classes | Replace with Dense(512) → Dense(256) head |
| Unfrozen BN layers shift feature distribution | Head receives drifting inputs during fine-tuning | BatchNormalization after each Dense layer |
| More parameters → overfitting risk | Wider head needs stronger regularisation | Increase Dropout 0.3 → 0.4 |
| Animal classes dominate errors (pose + lighting) | Augmentation does not cover orientation or contrast variance | Add RandomRotation(0.1) + RandomContrast(0.2) |
| Unfreezing risks catastrophic forgetting | Large LR destroys pretrained weights in first few batches | Reduce lr: 1e-3 → 1e-4 |

---

### 4. Enhanced Model

```
Input (32×32×3)
→ Enhanced augmentation  (+ RandomRotation, RandomContrast)
→ Rescaling × 255
→ DenseNetPreprocess
→ Resizing (224×224)
→ DenseNet121            (denseblock4 unfrozen, training=True)
→ Dense(512, ReLU) + BatchNorm + Dropout(0.4)
→ Dense(256, ReLU) + BatchNorm + Dropout(0.3)
→ Dense(10, Softmax)
```

| Metric | Baseline | Enhanced | Δ |
|---|---|---|---|
| Test accuracy | 90.28% | **93.32%** | +3.04 pp |
| Test loss | 0.2919 | **0.2081** | −0.0838 |
| Macro F1 | 0.90 | **0.93** | +0.03 |
| Train/val gap | ~2.3% | ~0.5% | Tighter |
| Cat F1 | 0.81 | **0.87** | +0.06 |
| Bird F1 | 0.89 | **0.93** | +0.04 |

---

### 5. Per-class Results (Enhanced Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 0.94 | 0.95 | 0.95 |
| automobile | 0.96 | 0.96 | 0.96 |
| bird | 0.93 | 0.92 | 0.93 |
| cat | 0.90 | 0.85 | **0.87** ← hardest |
| deer | 0.94 | 0.92 | 0.93 |
| dog | 0.88 | 0.91 | 0.89 |
| frog | 0.92 | 0.98 | 0.95 |
| horse | 0.96 | 0.95 | 0.95 |
| ship | 0.98 | 0.95 | 0.96 |
| truck | 0.93 | 0.97 | 0.95 |

Cat remains the hardest class — cats and dogs share body shape, fur texture, and four-legged pose at 32×32 resolution, making the cat/dog boundary the most structurally difficult discrimination in CIFAR-10.

---

## Key Technical Decisions

### Why DenseNet121

DenseNet121 was selected over ResNet50, InceptionV3, and MobileNetV2 for three reasons directly relevant to this dataset:

1. **Feature reusability** — every layer receives all preceding feature maps via concatenation, giving the classifier simultaneous access to low-level textures and high-level object identity. Critical for separating visually similar classes at 32×32.
2. **Gradient flow** — dense connections provide direct gradient highways to all layers, making training a 121-layer network on 40,000 samples stable without gradient collapse.
3. **Parameter efficiency** — ~8M parameters vs ResNet50's ~25M. With only 40,000 training images, fewer parameters means lower overfitting risk without sacrificing representational capacity.

### Why Unfreeze Only Block 4

Blocks 1–3 detect universal features (edges, textures, gradients) that transfer directly from ImageNet to CIFAR-10 — no adaptation needed. Block 4 detects whole-object identity calibrated for sharp 224×224 images. CIFAR-10's blurry upsampled inputs produce weak activations in frozen block 4. Unfreezing only block 4 re-calibrates the task-specific features while preserving the universal ones.

### Catastrophic Forgetting Prevention

At `lr=1e-3`, a gradient of 0.5 produces a weight update of 0.0005 per step — sufficient to push pretrained weights out of their learned region within 10 batches. At `lr=1e-4`, the update is 0.00005 — small enough to keep weights near their pretrained values while nudging them toward CIFAR-10. The training curves confirm this: the enhanced model's validation accuracy crosses above the baseline ceiling at epoch 10, precisely when block 4 completes re-calibration.

---

## Training Configuration

| Parameter | Baseline | Enhanced |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 |
| Batch size | 64 | 64 |
| Max epochs | 30 | 30 |
| Early stopping | patience=5 | patience=5 |
| LR reduction | factor=0.5, patience=3 | factor=0.5, patience=3 |
| Backbone | Fully frozen | Block 4 unfrozen |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install tensorflow tensorflow-datasets scikit-learn numpy matplotlib

# Open the notebook
jupyter notebook notebook.ipynb
```

> **GPU recommended.** Training runs on CPU but is significantly slower. In Google Colab: Runtime → Change runtime type → GPU.

To skip retraining and load saved checkpoints directly:

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import preprocess_input

@keras.saving.register_keras_serializable()
class DenseNetPreprocess(layers.Layer):
    def call(self, inputs):
        return preprocess_input(inputs)

baseline_model = keras.models.load_model("densenet121_baseline_best.keras")
enhanced_model = keras.models.load_model("densenet121_enhanced_best.keras")
```

---

## Dependencies

| Library | Version |
|---|---|
| Python | 3.10+ |
| TensorFlow / Keras | 2.15+ |
| tensorflow-datasets | 4.9+ |
| scikit-learn | 1.3+ |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |

---

## Relevance to Quantitative / ML Research Roles

The core practices in this project map directly to systematic model development in data-driven research:

- **Evidence-driven iteration** — every architectural change was motivated by a specific signal in the evaluation metrics, not intuition. This mirrors the backtest → diagnose → refine cycle in strategy development.
- **Robust evaluation** — test set kept completely held-out; all tuning decisions made on validation set only. Prevents the equivalent of overfitting to historical data in backtesting.
- **Pipeline reliability** — the preprocessing serialisation bug (Lambda layer silently dropping at inference) demonstrates why production ML pipelines require end-to-end testing, not just training-time validation.
- **Quantitative diagnosis** — confusion matrix analysis identified specific class-pair failure modes (cat/dog: 153 errors, bird/frog: 46 errors) and drove targeted fixes, analogous to attributing strategy drawdowns to specific market regimes.

---

*Assignment 2 — Deep Learning for AI | CNN Architectures*
