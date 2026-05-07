# MambaBTS: Brain Tumor Segmentation with State Space Models

> Based on the 2025 review paper:  
> *"Deep learning for brain tumor segmentation in multimodal MRI images"*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Part 1 — Image Processing Pipeline](#2-part-1--image-processing-pipeline)
3. [Part 2 — Model Architecture (MambaBTS)](#3-part-2--model-architecture-mambabts)
4. [Part 3 — Evaluation Metrics](#4-part-3--evaluation-metrics)
5. [Step-by-Step Code Walkthrough](#5-step-by-step-code-walkthrough)
6. [How to Run](#6-how-to-run)
7. [BraTS Dataset Format](#7-brats-dataset-format)
8. [References](#8-references)

---

## 1. Project Overview

Brain tumor segmentation is a critical step in surgical planning and treatment monitoring.  
MambaBTS replaces the standard Transformer attention modules inside a U-Net backbone with **Selective State Space Model (SSM) blocks** (Mamba / S6), giving a better trade-off between long-range context modelling and computational efficiency compared to both CNNs and pure Transformers.

The model takes 4-modality MRI volumes as input and produces a voxel-wise segmentation map with 4 classes:

| Label | Region |
|-------|--------|
| 0 | Background |
| 1 | Necrotic / Non-Enhancing Tumour Core (NCR/NET) |
| 2 | Peritumoral Edema (ED) |
| 3 | Enhancing Tumour (ET) |

The three BraTS evaluation regions are derived from these labels:

| BraTS Region | Labels included |
|---|---|
| Whole Tumour (WT) | 1 + 2 + 3 |
| Tumour Core (TC) | 1 + 3 |
| Enhancing Tumour (ET) | 3 |

---

## 2. Part 1 — Image Processing Pipeline

Raw MRI volumes cannot be used directly for training because they suffer from:

- **Intensity bias**: the MRI scanner produces a smooth, spatially varying signal drift across the image.
- **Skull and background noise**: non-brain tissue inflates the dynamic range and biases normalisation statistics.
- **Inter-subject intensity variability**: the same tissue type may appear at different absolute intensities in different subjects or scanners.

Three preprocessing stages are applied in sequence to every modality:

---

### 2.1 N4ITK Bias Field Correction

**What it does:**  
Estimates and removes the slowly varying multiplicative bias field $B(x)$ that corrupts the true tissue intensity $I_0(x)$:

$$I_{\text{observed}}(x) = I_0(x) \cdot B(x) + \eta(x)$$

where $\eta(x)$ is Gaussian noise.

**How it works (N4ITK algorithm):**  
The algorithm operates in the log domain, where the multiplicative bias becomes additive:

$$\log I_{\text{obs}} = \log I_0 + \log B$$

It iteratively refines $\log B$ using a B-spline fitting step and a sharpening step that maximises the entropy of the intensity histogram. The "N4" name comes from *Non-uniform intensity Normalisation* (Nu-Correct) with 4 specific improvements over earlier versions.

**Implementation in code (`N4BiasFieldCorrection`):**  
- Uses **ANTsPy** (`ants.n4_bias_field_correction`) when available — this is the reference implementation.  
- Falls back to a **Gaussian-smoothing approximation** when ANTsPy is not installed: the bias field is estimated as a heavily blurred version of the log-intensity image, which is then subtracted.

**Why it matters:**  
Without bias correction, a CNN or SSM can learn spurious spatial intensity patterns (e.g., "bright on the left side = tumour") that do not generalise across scanners.

---

### 2.2 Skull Stripping

**What it does:**  
Produces a binary **brain mask** that labels every voxel as brain or non-brain (skull, scalp, air, CSF outside the cortex). This mask is then applied to all 4 modalities.

**How it works:**  
Three approaches are tried in priority order:

1. **ANTsPyNet** (`brain_extraction` with modality `"t1"`) — deep-learning based, most accurate.  
2. **Morphological fallback** (used in the code as the default):  
   a. Threshold at 10% of the image maximum to separate foreground from background.  
   b. Morphological closing (3 iterations) to fill thin gaps.  
   c. Binary hole filling to close internal cavities.  
   d. Largest connected component selection to remove spurious islands.  
   e. Slight erosion (2 iterations) to peel off the skull rim.

**Why it matters:**  
The skull has much higher signal than brain tissue in T1 images. Including it would distort Z-score statistics and cause the model to segment the skull as tumour.

---

### 2.3 Z-score Normalization

**What it does:**  
Shifts and scales each voxel intensity $v$ within the brain mask so that:

$$v_{\text{norm}} = \frac{v - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$

where $\mu_{\text{brain}}$ and $\sigma_{\text{brain}}$ are the mean and standard deviation computed **only over brain voxels**.

**Why compute statistics inside the mask?**  
Background voxels are (approximately) zero after skull stripping. Including them would pull $\mu$ towards zero and inflate $\sigma$, making the effective normalisation of brain voxels weaker.

**Why it matters:**  
Neural networks converge faster and more reliably when inputs have approximately zero mean and unit variance. Z-scoring also makes the model more robust to scanner-to-scanner intensity differences.

---

### Pipeline Summary

```
Raw NIfTI (.nii.gz)
        │
        ▼
  N4ITK Bias Field Correction  (per modality)
        │
        ▼
  Skull Stripping  (brain mask from T1ce → applied to all 4 modalities)
        │
        ▼
  Z-score Normalization  (inside brain mask, per modality)
        │
        ▼
  Stack to (4, H, W, D) tensor  →  ready for the model
```

---

## 3. Part 2 — Model Architecture (MambaBTS)

### 3.1 Why Not CNNs?

Convolutional Neural Networks (CNNs) such as the original U-Net use **local receptive fields**. A 3×3×3 convolution sees only a small neighbourhood of voxels at a time. Tumours are spatially diffuse — the edema (class 2) often surrounds the core (class 1) by many voxels, and the model needs to connect distant evidence to segment correctly. CNNs achieve this only through many stacked layers, which is expensive.

### 3.2 Why Not Transformers?

Vision Transformers (ViTs) and Swin Transformers use **self-attention**, which computes relationships between every pair of tokens. For a 3-D volume of size $H \times W \times D$ with sequence length $L = H \cdot W \cdot D$, the attention matrix has $O(L^2)$ size. For a 128³ volume, $L = 2{,}097{,}152$ — this is completely infeasible without aggressive windowing, which then re-introduces limited context.

### 3.3 Why Mamba (SSM)?

State Space Models process sequences with $O(L)$ time and memory complexity, matching CNNs in efficiency while modelling **global context** like Transformers. The key innovation in Mamba (S6) is **input-dependent (selective) parameters**: the SSM matrices $A$, $B$, $C$ and the step size $\Delta$ are all functions of the input, allowing the model to dynamically decide which information to retain in the hidden state.

| Method | Complexity | Global Context | Input-dependent |
|--------|------------|----------------|-----------------|
| CNN (U-Net) | $O(L)$ | ✗ | ✗ |
| Transformer | $O(L^2)$ | ✓ | ✓ |
| SSM (S4) | $O(L)$ | ✓ | ✗ |
| **Mamba (S6)** | **$O(L)$** | **✓** | **✓** |

---

### 3.4 The Selective SSM (S6)

The continuous-time SSM is defined by:

$$\dot{h}(t) = A\,h(t) + B\,x(t)$$
$$y(t) = C\,h(t) + D\,x(t)$$

where $h(t) \in \mathbb{R}^N$ is the hidden state, $x(t) \in \mathbb{R}^D$ is the input, and $y(t) \in \mathbb{R}^D$ is the output.

**Discretisation (Zero-Order Hold):**

$$\bar{A} = \exp(\Delta \cdot A), \qquad \bar{B} = \Delta \cdot B$$

$$h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \qquad y_t = C\,h_t + D\,x_t$$

**Selectivity** comes from making $\Delta$, $B$, and $C$ functions of the input $x_t$ via small linear projections. This means the model learns *when to forget* (through $\bar{A}$) and *what to write* to its state (through $\bar{B}$), conditioned on the current input.

---

### 3.5 The MambaBlock

Each `MambaBlock` wraps the SSM with:

1. **Layer Normalization** — pre-norm for training stability.
2. **Expand projection** — doubles the channel count to create a richer representation.
3. **Depth-wise 1-D convolution** — captures local context *before* the SSM, analogous to the local window in Swin Transformers.
4. **SiLU activation** — smooth gating non-linearity.
5. **Selective SSM (S6)** — global sequence modelling.
6. **SiLU gating branch** — multiplicative gate that controls information flow.
7. **Output projection** — projects back to the original channel count.
8. **Residual connection** — preserves low-level features.

---

### 3.6 MambaBlock3D — Adapting to Volumetric Data

MRI is inherently 3-D. Rather than processing all $H \times W \times D$ voxels as a single flat sequence (which would be enormous), `MambaBlock3D` applies three independent 1-D Mamba scans along each spatial axis:

- **Axial scan** — flattens $H \times W$ planes, scans along depth $D$
- **Coronal scan** — flattens $W \times D$ planes, scans along height $H$
- **Sagittal scan** — flattens $H \times D$ planes, scans along width $W$

The three outputs are summed and passed through a $1 \times 1 \times 1$ fusion convolution. This gives global context in all three anatomical directions without the $O(L^2)$ cost.

---

### 3.7 Overall MambaBTS Architecture

```
Input (B, 4, 128, 128, 128)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  ENCODER                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Level 1: ConvBlock(4→32)  + MambaBlock3D  → skip_1     │  │
│  │          MaxPool3D                                       │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 2: ConvBlock(32→64) + MambaBlock3D  → skip_2     │  │
│  │          MaxPool3D                                       │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 3: ConvBlock(64→128)+ MambaBlock3D  → skip_3     │  │
│  │          MaxPool3D                                       │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 4: ConvBlock(128→256)+MambaBlock3D  → skip_4     │  │
│  │          MaxPool3D                                       │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  BOTTLENECK                     │
│  ConvBlock(256→512)             │
│  MambaBlock3D                   │
└─────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  DECODER                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Level 4: ConvTranspose(512→256) + cat(skip_4)          │  │
│  │          ConvBlock(512→256) + MambaBlock3D              │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 3: ConvTranspose(256→128) + cat(skip_3)          │  │
│  │          ConvBlock(256→128) + MambaBlock3D              │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 2: ConvTranspose(128→64)  + cat(skip_2)          │  │
│  │          ConvBlock(128→64)  + MambaBlock3D              │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Level 1: ConvTranspose(64→32)   + cat(skip_1)          │  │
│  │          ConvBlock(64→32)   + MambaBlock3D              │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Segmentation Head: Conv3d(32→4, kernel=1)
        │
        ▼
Output (B, 4, 128, 128, 128)  — logits per class
```

**Skip connections** carry the Mamba-refined encoder features directly to the corresponding decoder level, preserving fine spatial detail that the bottleneck may have compressed away.

---

## 4. Part 3 — Evaluation Metrics

### 4.1 Dice Similarity Coefficient (DSC)

The DSC (also called the F1 score in set-theoretic terms) measures the overlap between the predicted segmentation $P$ and the ground truth $G$:

$$\text{DSC} = \frac{2 |P \cap G|}{|P| + |G|}$$

- **Range:** 0 (no overlap) to 1 (perfect overlap).
- **Sensitivity to class imbalance:** the denominator uses the *sum* of set sizes, not their union, which makes DSC more forgiving of small structures than Intersection-over-Union (IoU).
- **BraTS standard:** report DSC separately for WT, TC, and ET.

**Implementation (`dice_similarity_coefficient`):**  
The function computes:
- Per-class DSC (classes 1, 2, 3)
- WT DSC: union of all tumour voxels
- TC DSC: labels 1 and 3
- ET DSC: label 3 only
- Mean DSC over WT, TC, ET (the primary BraTS ranking metric)

During training, a *soft* differentiable Dice loss is used:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_v p_v \cdot g_v + \epsilon}{\sum_v p_v + \sum_v g_v + \epsilon}$$

where $p_v$ are the softmax probabilities and $g_v$ are the one-hot ground truth values. This is combined with Cross-Entropy loss for stable gradients near the decision boundary:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{CE}}$$

---

### 4.2 Hausdorff Distance (HD95)

The Hausdorff Distance measures the **worst-case boundary error** between two segmentation surfaces:

$$H(P, G) = \max\!\left(\sup_{p \in \partial P} \inf_{g \in \partial G} d(p,g),\;\; \sup_{g \in \partial G} \inf_{p \in \partial P} d(g,p)\right)$$

The standard HD is sensitive to single outlier voxels, so in practice the **95th percentile** (HD95) is used:

$$\text{HD95} = P_{95}\!\left(\{d(p, \partial G) : p \in \partial P\} \cup \{d(g, \partial P) : g \in \partial G\}\right)$$

- **Unit:** millimetres (physical space, using voxel spacing).
- **Range:** 0 mm (perfect boundary match) to a large value for gross segmentation errors.
- **Complement to DSC:** DSC penalises volume errors; HD95 penalises boundary placement errors. A model can have high DSC but poor HD95 if it produces a slightly shifted but otherwise correct mask.

**Implementation (`hausdorff_distance_95`):**  
Uses `scipy.ndimage.distance_transform_edt` to compute exact Euclidean distances from each surface voxel to the opposing surface, scaled by voxel spacing. The 95th percentile of the combined distances gives HD95.

---

## 5. Step-by-Step Code Walkthrough

### Step 1 — Preprocessing a Subject

```python
from mamba_bts_impl import BraTSPreprocessor

preprocessor = BraTSPreprocessor()

modality_paths = {
    "t1":    "subject_001/subject_001_t1.nii.gz",
    "t1ce":  "subject_001/subject_001_t1ce.nii.gz",
    "t2":    "subject_001/subject_001_t2.nii.gz",
    "flair": "subject_001/subject_001_flair.nii.gz",
}

volume, seg = preprocessor.preprocess_subject(
    modality_paths,
    seg_path="subject_001/subject_001_seg.nii.gz",
)
# volume.shape → (4, 240, 240, 155)
# seg.shape    → (240, 240, 155)
```

Internally, the call:
1. Loads each NIfTI file and converts to `float32`.
2. Applies `N4BiasFieldCorrection.correct()` to each modality independently.
3. Calls `SkullStripper.strip()` on the corrected T1ce volume to obtain `brain_mask`.
4. Applies `ZScoreNormalization.normalize()` to each modality using `brain_mask`.
5. Stacks the 4 normalised volumes into a `(4, H, W, D)` array.

---

### Step 2 — Creating a Dataset

```python
from mamba_bts_impl import BraTSDataset
from torch.utils.data import DataLoader

dataset = BraTSDataset(
    root="path/to/BraTS2024/",
    patch_size=(128, 128, 128),
    preprocess=True,
)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

batch = next(iter(loader))
print(batch["image"].shape)  # (1, 4, 128, 128, 128)
print(batch["seg"].shape)    # (1, 128, 128, 128)
```

`BraTSDataset.__getitem__` applies preprocessing, then centre-crops or zero-pads each volume to `patch_size`. This ensures all batches have the same shape.

---

### Step 3 — Building the Model

```python
import torch
from mamba_bts_impl import MambaBTS

model = MambaBTS(
    in_channels=4,       # T1, T1ce, T2, FLAIR
    num_classes=4,       # background + NCR + ED + ET
    base_features=32,    # feature channels at level 1
    d_state=16,          # SSM state dimension N
)

x = torch.randn(1, 4, 128, 128, 128)
logits = model(x)        # (1, 4, 128, 128, 128)
pred   = logits.argmax(dim=1)  # (1, 128, 128, 128)
```

The model contains no Transformer attention layers — every contextual operation goes through `MambaBlock3D`, which is linear in the number of voxels.

---

### Step 4 — Training

```python
from mamba_bts_impl import train

trained_model = train(
    data_root="path/to/BraTS2024/",
    num_epochs=100,
    batch_size=1,
    lr=1e-4,
    patch_size=(128, 128, 128),
    base_features=32,
    d_state=16,
    checkpoint_dir="checkpoints/",
    device_str="auto",  # picks CUDA > MPS > CPU
)
```

The training loop:
1. Runs `train_one_epoch` — forward pass, combined loss, backward pass with gradient clipping (`max_norm=1.0`), optimiser step.
2. Uses **AdamW** (weight decay $10^{-5}$) + **Cosine Annealing LR** (min LR $10^{-6}$).
3. Evaluates on a 10% validation split every 10 epochs using DSC + HD95.
4. Saves the best checkpoint (`best_model.pth`) based on mean DSC over WT/TC/ET.
5. Uses **AMP (automatic mixed precision)** on CUDA for memory efficiency.

---

### Step 5 — Evaluating the Model

```python
from mamba_bts_impl import dice_similarity_coefficient, hausdorff_distance_95
import numpy as np

pred   = np.load("prediction.npy").astype(np.uint8)   # (H, W, D)
target = np.load("ground_truth.npy").astype(np.uint8)  # (H, W, D)

dsc  = dice_similarity_coefficient(pred, target, num_classes=4)
hd95 = hausdorff_distance_95(pred, target, num_classes=4)

print(f"WT DSC:  {dsc['WT']:.4f}")
print(f"TC DSC:  {dsc['TC']:.4f}")
print(f"ET DSC:  {dsc['ET']:.4f}")
print(f"WT HD95: {hd95['WT']:.2f} mm")
print(f"TC HD95: {hd95['TC']:.2f} mm")
print(f"ET HD95: {hd95['ET']:.2f} mm")
```

---

### Step 6 — Running Inference on a New Subject

```python
from mamba_bts_impl import run_inference

pred = run_inference(
    checkpoint_path="checkpoints/best_model.pth",
    modality_paths={
        "t1":    "new_subject/t1.nii.gz",
        "t1ce":  "new_subject/t1ce.nii.gz",
        "t2":    "new_subject/t2.nii.gz",
        "flair": "new_subject/flair.nii.gz",
    },
    output_path="new_subject/prediction.nii.gz",
)
```

The prediction is saved as a NIfTI file, inheriting the affine matrix and header of the input T1 volume so it overlays correctly in ITK-SNAP or 3D Slicer.

---

## 6. How to Run

### Prerequisites

```bash
pip install torch torchvision nibabel scipy scikit-learn
# Optional (highly recommended for accurate preprocessing):
pip install antspyx antspynet
```

### Quick Sanity Check (no dataset needed)

```bash
python mamba_bts_impl.py smoke-test --device cpu
```

Expected output:
```
============================================================
MambaBTS Smoke Test
============================================================

[1/4] Instantiating MambaBTS model...
  Trainable parameters: X,XXX,XXX

[2/4] Forward pass on random input...
  Input  shape: (1, 4, 64, 64, 64)
  Output shape: (1, 4, 64, 64, 64)  ✓

[3/4] Loss computation (Dice + CE)...
  Combined loss: X.XXXX  ✓

[4/4] Evaluation metrics (DSC + HD95)...
  DSC results: ...
  HD95 results: ...

============================================================
Smoke test PASSED ✓
============================================================
```

### Training

```bash
python mamba_bts_impl.py train \
    --data-root /path/to/BraTS2024 \
    --epochs 100 \
    --batch-size 1 \
    --patch-size 128 128 128 \
    --base-features 32 \
    --d-state 16 \
    --checkpoint-dir checkpoints/ \
    --device auto
```

### Inference

```bash
python mamba_bts_impl.py infer \
    --checkpoint checkpoints/best_model.pth \
    --t1    subject/t1.nii.gz \
    --t1ce  subject/t1ce.nii.gz \
    --t2    subject/t2.nii.gz \
    --flair subject/flair.nii.gz \
    --output subject/pred.nii.gz
```

---

## 7. BraTS Dataset Format

The BraTS (Brain Tumour Segmentation) challenge provides multi-institutional, multi-modal MRI scans pre-processed to:
- Common anatomical template (SRI24)
- 1 mm isotropic resolution
- Skull-stripped (already — our pipeline adds a re-strip for safety)

Download: [https://www.synapse.org/brats](https://www.synapse.org/brats)

Expected directory structure:

```
BraTS2024/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   └── BraTS-GLI-00000-000-seg.nii.gz
├── BraTS-GLI-00001-000/
│   └── ...
```

The `BraTSDataset` class auto-discovers subjects from this layout.

---

## 8. References

1. Gu, A., & Dao, T. (2023). **Mamba: Linear-time sequence modeling with selective state spaces.** *arXiv:2312.00752*.
2. Isensee, F., et al. (2021). **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** *Nature Methods, 18*, 203–211.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional networks for biomedical image segmentation.** *MICCAI 2015*.
4. Tustison, N. J., et al. (2010). **N4ITK: Improved N3 bias correction.** *IEEE TMI, 29*(6), 1310–1320.
5. Menze, B. H., et al. (2015). **The multimodal brain tumor image segmentation benchmark (BRATS).** *IEEE TMI, 34*(10), 1993–2024.
6. Liu, Y., et al. (2025). **Deep learning for brain tumor segmentation in multimodal MRI images.** *(2025 review paper)*.
