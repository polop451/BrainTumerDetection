"""
MambaBTS: State Space Model-based Brain Tumor Segmentation
Based on: "Deep learning for brain tumor segmentation in multimodal MRI images" (2025)

Dataset: BraTS format — 4 MRI modalities: T1, T1ce, T2, FLAIR
Labels:  0=Background, 1=Necrotic/Non-Enhancing Tumor Core (NCR/NET),
         2=Peritumoral Edema (ED), 3=Enhancing Tumor (ET)

Requirements:
    pip install torch torchvision nibabel scipy scikit-learn antspyx
    (antspyx is optional; placeholder logic is used if unavailable)
"""

from __future__ import annotations

import math
import os
import warnings
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMAGE PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


class N4BiasFieldCorrection:
    """
    N4ITK Bias Field Correction.

    Attempts to use ANTsPy (full implementation). Falls back to a
    Gaussian-smoothing-based approximation when ANTsPy is unavailable.
    """

    def __init__(self, n_iterations: List[int] = [50, 50, 50, 50],
                 convergence_threshold: float = 1e-6):
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self._use_ants = self._check_ants()

    @staticmethod
    def _check_ants() -> bool:
        try:
            import ants  # noqa: F401
            return True
        except ImportError:
            return False

    def correct(self, image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply N4ITK bias field correction.

        Args:
            image: 3-D float array (H, W, D).
            mask:  Binary brain mask (optional). If None, a threshold mask is
                   derived automatically.
        Returns:
            Bias-field-corrected image with the same shape and dtype.
        """
        if self._use_ants:
            return self._correct_ants(image, mask)
        return self._correct_approx(image, mask)

    def _correct_ants(self, image: np.ndarray,
                      mask: Optional[np.ndarray]) -> np.ndarray:
        import ants
        ants_img = ants.from_numpy(image.astype(np.float32))
        ants_mask = (
            ants.from_numpy(mask.astype(np.float32))
            if mask is not None
            else ants.get_mask(ants_img)
        )
        corrected = ants.n4_bias_field_correction(
            ants_img,
            mask=ants_mask,
            convergence={"iters": self.n_iterations,
                         "tol": self.convergence_threshold},
        )
        return corrected.numpy()

    @staticmethod
    def _correct_approx(image: np.ndarray,
                        mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Approximation: estimate the bias field as a heavily blurred version of
        the log-intensity image, then divide it out (log-domain subtraction).
        This is not as accurate as the true N4 algorithm but is dependency-free.
        """
        from scipy.ndimage import gaussian_filter

        img = image.astype(np.float64)
        img = np.where(img > 0, img, np.finfo(np.float64).eps)

        log_img = np.log(img)
        # Estimate bias field with large-sigma smoothing
        bias_field = gaussian_filter(log_img, sigma=10)
        corrected = np.exp(log_img - bias_field)

        # Restore original intensity scale
        scale = np.median(img[img > 0]) / (np.median(corrected[corrected > 0]) + 1e-8)
        corrected = corrected * scale
        return corrected.astype(image.dtype)


class ZScoreNormalization:
    """
    Z-score (zero-mean, unit-variance) normalization computed inside the
    brain mask to avoid background voxels distorting the statistics.
    """

    def __init__(self, mask_threshold: float = 0.0):
        self.mask_threshold = mask_threshold

    def normalize(self, image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize a single modality volume.

        Args:
            image: 3-D float array.
            mask:  Binary mask selecting brain voxels. Derived from
                   `mask_threshold` if not supplied.
        Returns:
            Normalized array with zero mean and unit std (within mask).
        """
        if mask is None:
            mask = image > self.mask_threshold

        brain_voxels = image[mask]
        mean = brain_voxels.mean()
        std = brain_voxels.std() + 1e-8

        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[mask] = (image[mask] - mean) / std
        return normalized


class SkullStripper:
    """
    Skull Stripping — isolates brain tissue from the skull and background.

    Priority:
        1. ANTsPy  (HD-BET or antspynet BrainExtraction)
        2. FSL BET (via subprocess if on PATH)
        3. Intensity-threshold + morphological fallback
    """

    def __init__(self, method: str = "auto"):
        """
        Args:
            method: "auto" | "ants" | "fsl" | "threshold"
        """
        self.method = method

    def strip(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (stripped_image, brain_mask).

        The brain mask is a binary array with 1 inside the brain.
        """
        if self.method in ("auto", "ants") and self._check_ants():
            return self._strip_ants(image)
        if self.method in ("auto", "threshold"):
            return self._strip_threshold(image)
        raise ValueError(f"Unknown skull-stripping method: {self.method}")

    @staticmethod
    def _check_ants() -> bool:
        try:
            import antspynet  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _strip_ants(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import ants
        import antspynet

        ants_img = ants.from_numpy(image.astype(np.float32))
        prob = antspynet.brain_extraction(ants_img, modality="t1")
        mask = (prob.numpy() > 0.5).astype(np.uint8)
        stripped = image * mask
        return stripped.astype(np.float32), mask

    @staticmethod
    def _strip_threshold(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple morphological skull-stripping used when no external library
        is available.  Works well for pre-processed BraTS volumes.
        """
        from scipy.ndimage import binary_closing, binary_erosion, generate_binary_structure

        # Otsu-like threshold (foreground = voxels above 10% of max)
        threshold = image.max() * 0.1
        binary = (image > threshold).astype(bool)

        # Fill holes and keep the largest connected component
        struct = generate_binary_structure(3, 2)
        closed = binary_closing(binary, structure=struct, iterations=3)
        filled = binary_fill_holes(closed)

        labeled, n_components = label(filled)
        if n_components == 0:
            return image.astype(np.float32), filled.astype(np.uint8)

        # Retain largest component
        sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
        largest = np.argmax(sizes) + 1
        brain_mask = (labeled == largest).astype(np.uint8)

        # Slight erosion to remove skull rim
        brain_mask = binary_erosion(brain_mask, structure=struct,
                                    iterations=2).astype(np.uint8)
        stripped = image * brain_mask
        return stripped.astype(np.float32), brain_mask


class BraTSPreprocessor:
    """
    Full preprocessing pipeline for a single BraTS subject.

    Steps applied to each of the 4 modalities:
        1. N4ITK Bias Field Correction
        2. Skull Stripping  (mask derived from T1ce, applied to all modalities)
        3. Z-score Normalization (within brain mask)
    """

    def __init__(self,
                 n4_iterations: List[int] = [50, 50, 50, 50],
                 skull_strip_method: str = "auto"):
        self.n4 = N4BiasFieldCorrection(n_iterations=n4_iterations)
        self.skull_stripper = SkullStripper(method=skull_strip_method)
        self.normalizer = ZScoreNormalization()

    def preprocess_subject(
        self,
        modality_paths: dict,          # {"t1": path, "t1ce": path, "t2": path, "flair": path}
        seg_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load, correct, strip and normalise all modalities.

        Returns:
            volume : float32 ndarray of shape (4, H, W, D)
            seg    : uint8   ndarray of shape (H, W, D)  or None
        """
        modality_order = ["t1", "t1ce", "t2", "flair"]
        volumes = {}

        # ── Step 1: load & N4 correct ────────────────────────────────────────
        for mod in modality_order:
            img_nii = nib.load(modality_paths[mod])
            data = img_nii.get_fdata(dtype=np.float32)
            corrected = self.n4.correct(data)
            volumes[mod] = corrected

        # ── Step 2: skull stripping (reference modality = T1ce) ──────────────
        _, brain_mask = self.skull_stripper.strip(volumes["t1ce"])

        # ── Step 3: Z-score normalisation ────────────────────────────────────
        processed = []
        for mod in modality_order:
            norm = self.normalizer.normalize(volumes[mod],
                                             mask=brain_mask.astype(bool))
            processed.append(norm)

        volume = np.stack(processed, axis=0)   # (4, H, W, D)

        seg = None
        if seg_path is not None:
            seg_nii = nib.load(seg_path)
            seg = seg_nii.get_fdata(dtype=np.float32).astype(np.uint8)

        return volume, seg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MAMBA BLOCKS & MAMBABTS ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — the core of the Mamba block.

    Implements input-dependent (selective) A, B, C projections so the model
    can decide which information to propagate through time, unlike fixed-
    parameter SSMs (S4).  The discretisation uses the Zero-Order Hold (ZOH)
    method.
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # ── Input-selective projections ───────────────────────────────────────
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # ── Learnable SSM parameters ──────────────────────────────────────────
        # A: initialised with HiPPO-like values (log-space for positivity)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # ── Initialise dt_proj bias for stable gradients ─────────────────────
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)  — sequence along spatial dimension L.
        Returns:
            y: (B, L, d_model)
        """
        B, L, D = x.shape
        N = self.d_state

        # ── Compute input-dependent Δ, B, C ───────────────────────────────────
        x_proj = self.x_proj(x)                              # (B, L, dt_rank + 2N)
        dt_raw, B_mat, C_mat = x_proj.split(
            [self.dt_rank, N, N], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))                # (B, L, D)

        # ── SSM parameters ────────────────────────────────────────────────────
        A = -torch.exp(self.A_log.float())                   # (D, N)

        # ── Sequential SSM scan (per-step to avoid (B,L,D,N) allocation) ─────
        # Pre-allocating dA/dB as (B, L, D, N) would require ~GB of VRAM for
        # the sequence lengths produced by 96^3 patches.  Instead we slice
        # dt, B_mat, C_mat one timestep at a time — same math, O(1) overhead.
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)                    # (B, D, 1)
            dA_t = torch.exp(dt_t * A)                       # (B, D, N)
            dB_t = dt_t * B_mat[:, t].unsqueeze(1)           # (B, D, N)
            h = dA_t * h + dB_t * x[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1) # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                           # (B, L, D)
        y = y + x * self.D                                    # skip via D
        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block with:
        - Layer normalisation
        - Input/output projections + SiLU gating
        - Depth-wise 1-D convolution for local context
        - Selective SSM (S6)
        - Residual connection
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dt_rank: Optional[int] = None):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)

        # ── Expand projection ─────────────────────────────────────────────────
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # ── Short 1-D conv for local context before SSM ───────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # ── Expand and split into main & gate branches ────────────────────────
        xz = self.in_proj(x)                                 # (B, L, 2·d_inner)
        x_main, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)

        # ── Local depthwise conv (operates along L) ───────────────────────────
        x_conv = self.conv1d(x_main.transpose(1, 2))        # (B, d_inner, L + pad)
        x_conv = x_conv[..., :x_main.shape[1]]              # trim padding
        x_conv = x_conv.transpose(1, 2)                     # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # ── Selective SSM ─────────────────────────────────────────────────────
        y = self.ssm(x_conv)                                 # (B, L, d_inner)

        # ── Gating ───────────────────────────────────────────────────────────
        y = y * F.silu(z)

        out = self.out_proj(y)                               # (B, L, d_model)
        return out + residual


class MambaBlock3D(nn.Module):
    """
    Adapts MambaBlock for 3-D volumetric feature maps by serialising the
    spatial dimensions into a sequence, applying the Mamba block, and then
    reshaping back.

    Three scan directions (axial/coronal/sagittal) are fused by averaging to
    capture volumetric context from multiple perspectives.
    """

    def __init__(self, channels: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.channels = channels
        # One Mamba block per scan direction
        self.mamba_z = MambaBlock(channels, d_state=d_state,
                                  d_conv=d_conv, expand=expand)   # axial (D)
        self.mamba_y = MambaBlock(channels, d_state=d_state,
                                  d_conv=d_conv, expand=expand)   # coronal (H)
        self.mamba_x = MambaBlock(channels, d_state=d_state,
                                  d_conv=d_conv, expand=expand)   # sagittal (W)
        self.fusion = nn.Conv3d(channels, channels,
                                kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D)
        Returns:
            (B, C, H, W, D)
        """
        B, C, H, W, D = x.shape

        # ── Axial scan: flatten H×W, scan along D ────────────────────────────
        xz = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, D, C)
        yz = self.mamba_z(xz).reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

        # ── Coronal scan: flatten W×D, scan along H ───────────────────────────
        xy = x.permute(0, 3, 4, 1, 2).reshape(B * W * D, H, C)
        yy = self.mamba_y(xy).reshape(B, W, D, H, C).permute(0, 4, 3, 1, 2)

        # ── Sagittal scan: flatten H×D, scan along W ──────────────────────────
        xx = x.permute(0, 2, 4, 1, 3).reshape(B * H * D, W, C)
        yx = self.mamba_x(xx).reshape(B, H, D, W, C).permute(0, 4, 1, 3, 2)

        fused = self.fusion(yz + yy + yx)
        return F.relu(self.norm(fused))


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """Standard 3-D double-conv block with GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder stage: ConvBlock → MambaBlock3D → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, d_state: int = 16):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.mamba = MambaBlock3D(out_ch, d_state=d_state)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.mamba(self.conv(x))   # feature map kept for skip connection
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Decoder stage: Upsample → concat skip → ConvBlock → MambaBlock3D."""

    def __init__(self, in_ch: int, out_ch: int, d_state: int = 16):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch * 2, out_ch)
        self.mamba = MambaBlock3D(out_ch, d_state=d_state)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims in case of odd input sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.mamba(self.conv(x))


class MambaBTS(nn.Module):
    """
    MambaBTS: U-Net shaped Brain Tumor Segmentation network with Mamba (S6)
    blocks replacing self-attention in both encoder and decoder paths.

    Architecture:
        Input  : (B, 4, H, W, D)   — 4 MRI modalities
        Encoder: 4 levels (channels: 32 → 64 → 128 → 256)
        Bridge : Mamba3D bottleneck (512 channels)
        Decoder: 4 levels (channels: 256 → 128 → 64 → 32)
        Output : (B, num_classes, H, W, D)

    Skip connections carry Mamba-refined features from each encoder level.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 base_features: int = 32, d_state: int = 16):
        super().__init__()

        f = base_features                         # 32, 64, 128, 256, 512

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, f,       d_state=d_state)
        self.enc2 = EncoderBlock(f,           f * 2,   d_state=d_state)
        self.enc3 = EncoderBlock(f * 2,       f * 4,   d_state=d_state)
        self.enc4 = EncoderBlock(f * 4,       f * 8,   d_state=d_state)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBlock3D(f * 8, f * 16),
            MambaBlock3D(f * 16, d_state=d_state),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(f * 16, f * 8,  d_state=d_state)
        self.dec3 = DecoderBlock(f * 8,  f * 4,  d_state=d_state)
        self.dec2 = DecoderBlock(f * 4,  f * 2,  d_state=d_state)
        self.dec1 = DecoderBlock(f * 2,  f,      d_state=d_state)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H, W, D)
        Returns:
            logits: (B, num_classes, H, W, D)
        """
        # ── Encoder path ─────────────────────────────────────────────────────
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decoder path ─────────────────────────────────────────────────────
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.seg_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATASET
# ─────────────────────────────────────────────────────────────────────────────


class BraTSDataset(Dataset):
    """
    BraTS-format dataset loader.

    Expected directory layout:
        root/
          subject_001/
            subject_001_t1.nii.gz
            subject_001_t1ce.nii.gz
            subject_001_t2.nii.gz
            subject_001_flair.nii.gz
            subject_001_seg.nii.gz   (optional for inference)
          subject_002/
            ...

    The constructor builds the subject list automatically from `root`.
    Preprocessing (N4 + skull strip + Z-score) is applied on-the-fly.
    """

    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    def __init__(self, root: str,
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 preprocess: bool = True,
                 augment: bool = False):
        self.root = root
        self.patch_size = patch_size
        self.augment = augment
        self.preprocessor = BraTSPreprocessor() if preprocess else None
        self.subjects = self._build_subject_list()

    def _build_subject_list(self) -> List[dict]:
        subjects = []
        for subj_dir in sorted(os.listdir(self.root)):
            full_path = os.path.join(self.root, subj_dir)
            if not os.path.isdir(full_path):
                continue
            entry = {"name": subj_dir}
            for mod in self.MODALITIES:
                fname = os.path.join(full_path, f"{subj_dir}_{mod}.nii.gz")
                if not os.path.exists(fname):
                    break
                entry[mod] = fname
            else:
                seg_path = os.path.join(full_path, f"{subj_dir}_seg.nii.gz")
                entry["seg"] = seg_path if os.path.exists(seg_path) else None
                subjects.append(entry)
        return subjects

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        subj = self.subjects[idx]
        modality_paths = {m: subj[m] for m in self.MODALITIES}

        if self.preprocessor is not None:
            volume, seg = self.preprocessor.preprocess_subject(
                modality_paths, seg_path=subj.get("seg")
            )
        else:
            # Load raw volumes without preprocessing
            vols = [nib.load(modality_paths[m]).get_fdata(dtype=np.float32)
                    for m in self.MODALITIES]
            volume = np.stack(vols, axis=0)
            seg = (nib.load(subj["seg"]).get_fdata(dtype=np.float32).astype(np.uint8)
                   if subj.get("seg") else None)

        # ── Crop / pad to patch size ──────────────────────────────────────────
        volume = self._resize_volume(volume)         # (4, H, W, D)
        if seg is not None:
            seg = self._resize_volume(seg[None])[0]  # (H, W, D)

        volume_tensor = torch.from_numpy(volume)
        result = {"image": volume_tensor, "name": subj["name"]}
        if seg is not None:
            result["seg"] = torch.from_numpy(seg).long()
        return result

    def _resize_volume(self, arr: np.ndarray) -> np.ndarray:
        """Centre-crop or zero-pad to reach `patch_size` on the last 3 axes."""
        target = self.patch_size
        spatial = arr.shape[-3:]
        slices_in, slices_out = [], []
        padded = np.zeros(arr.shape[:-3] + target, dtype=arr.dtype)

        for s, t in zip(spatial, target):
            if s >= t:
                start = (s - t) // 2
                slices_in.append(slice(start, start + t))
                slices_out.append(slice(0, t))
            else:
                pad = (t - s) // 2
                slices_in.append(slice(0, s))
                slices_out.append(slice(pad, pad + s))

        padded[..., slices_out[0], slices_out[1], slices_out[2]] = \
            arr[..., slices_in[0], slices_in[1], slices_in[2]]
        return padded


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss averaged across foreground classes."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5,
                 ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.start_class = 1 if ignore_background else 0

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D) — raw model output
            targets : (B, H, W, D)    — integer class labels
        """
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes)   # (B,H,W,D,C)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dice_scores = []
        for c in range(self.start_class, self.num_classes):
            p = probs[:, c]
            g = targets_onehot[:, c]
            intersection = (p * g).sum()
            dice = (2 * intersection + self.smooth) / \
                   (p.sum() + g.sum() + self.smooth)
            dice_scores.append(dice)

        return 1.0 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    """Dice Loss + Cross-Entropy, weighted sum."""

    def __init__(self, num_classes: int = 4, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────


def dice_similarity_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> dict:
    """
    Compute per-class and mean Dice Similarity Coefficient (DSC).

    BraTS evaluation regions:
        WT (Whole Tumour)   = labels {1, 2, 3}
        TC (Tumour Core)    = labels {1, 3}
        ET (Enhancing Tumour) = label {3}

    Args:
        pred   : 3-D integer array (H, W, D)
        target : 3-D integer array (H, W, D)
    Returns:
        dict with per-class DSC and the three BraTS standard region DSCs.
    """
    results = {}
    smooth = 1e-8
    start = 1 if ignore_background else 0

    for c in range(start, num_classes):
        p = (pred == c).astype(float)
        g = (target == c).astype(float)
        intersection = (p * g).sum()
        dsc = (2 * intersection + smooth) / (p.sum() + g.sum() + smooth)
        results[f"class_{c}"] = float(dsc)

    # ── Standard BraTS regions ────────────────────────────────────────────────
    def _dsc(p_mask, g_mask):
        i = (p_mask & g_mask).sum()
        return float((2 * i + smooth) / (p_mask.sum() + g_mask.sum() + smooth))

    results["WT"] = _dsc(pred >= 1, target >= 1)
    results["TC"] = _dsc(np.isin(pred, [1, 3]), np.isin(target, [1, 3]))
    results["ET"] = _dsc(pred == 3, target == 3)
    results["mean_dsc"] = float(np.mean([results["WT"],
                                          results["TC"],
                                          results["ET"]]))
    return results


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    percentile: float = 95.0,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """
    Compute the 95th-percentile Hausdorff Distance (HD95) per class and for
    the three BraTS standard regions.

    Args:
        pred          : integer label map (H, W, D)
        target        : integer label map (H, W, D)
        percentile    : percentile for Hausdorff (default 95)
        voxel_spacing : physical voxel size in mm (default isotropic 1 mm)
    Returns:
        dict with HD95 values in mm.
    """
    from scipy.ndimage import distance_transform_edt

    results = {}
    spacing = np.array(voxel_spacing)

    def _hd95(p_mask: np.ndarray, g_mask: np.ndarray) -> float:
        if p_mask.sum() == 0 and g_mask.sum() == 0:
            return 0.0
        if p_mask.sum() == 0 or g_mask.sum() == 0:
            return float("inf")
        # Surface voxels: where the binary mask meets background
        p_surface = p_mask ^ binary_fill_holes(p_mask)
        g_surface = g_mask ^ binary_fill_holes(g_mask)

        # Distance transform from each surface
        dt_p = distance_transform_edt(~p_surface, sampling=spacing)
        dt_g = distance_transform_edt(~g_surface, sampling=spacing)

        d_p2g = dt_g[p_surface]     # distances from pred surface to GT
        d_g2p = dt_p[g_surface]     # distances from GT surface to pred

        all_dist = np.concatenate([d_p2g, d_g2p])
        return float(np.percentile(all_dist, percentile))

    for c in range(1, num_classes):
        results[f"class_{c}"] = _hd95(pred == c, target == c)

    results["WT"] = _hd95(pred >= 1, target >= 1)
    results["TC"] = _hd95(np.isin(pred, [1, 3]), np.isin(target, [1, 3]))
    results["ET"] = _hd95(pred == 3, target == 3)
    return results


def evaluate_batch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> dict:
    """
    Run inference on a DataLoader and aggregate DSC + HD95 metrics.

    Returns:
        dict with mean DSC and HD95 across the dataset.
    """
    model.eval()
    all_dsc, all_hd95 = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)           # (B, 4, H, W, D)
            segs = batch.get("seg")

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()   # (B, H, W, D)

            if segs is not None:
                segs_np = segs.cpu().numpy()
                for b in range(preds.shape[0]):
                    dsc = dice_similarity_coefficient(preds[b], segs_np[b],
                                                      num_classes)
                    hd = hausdorff_distance_95(preds[b], segs_np[b],
                                               num_classes)
                    all_dsc.append(dsc)
                    all_hd95.append(hd)

    if not all_dsc:
        return {}

    mean_dsc = {k: float(np.mean([d[k] for d in all_dsc if k in d]))
                for k in all_dsc[0]}
    mean_hd95 = {k: float(np.mean([h[k] for h in all_hd95
                                   if k in h and not np.isinf(h[k])]))
                 for k in all_hd95[0]}

    return {"DSC": mean_dsc, "HD95": mean_hd95}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
) -> float:
    try:
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}", unit="batch",
                    dynamic_ncols=True, leave=False)
    except ImportError:
        pbar = dataloader

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in pbar:
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, segs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, segs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def train(
    data_root: str,
    num_epochs: int = 100,
    batch_size: int = 1,
    lr: float = 1e-4,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    num_classes: int = 4,
    base_features: int = 32,
    d_state: int = 16,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "auto",
):
    """
    End-to-end training entry point.

    Args:
        data_root     : path to BraTS dataset root.
        num_epochs    : total training epochs.
        batch_size    : mini-batch size (1 recommended for volumetric data).
        lr            : initial learning rate.
        patch_size    : spatial crop size (H, W, D).
        num_classes   : segmentation classes (4 for BraTS).
        base_features : base channel count for MambaBTS.
        d_state       : SSM state dimension.
        checkpoint_dir: directory where model checkpoints are saved.
        device_str    : "auto" | "cuda" | "mps" | "cpu"
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Env cleanup: remove CUDA_LAUNCH_BLOCKING if set by a prior smoke test ─
    # It forces synchronous kernel execution and prevents DataParallel from
    # overlapping work across multiple GPUs.
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Device selection ──────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[Train] Using device: {device}")

    # ── Dataset & Dataloader ──────────────────────────────────────────────────
    dataset = BraTSDataset(data_root, patch_size=patch_size,
                           preprocess=True, augment=True)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MambaBTS(in_channels=4, num_classes=num_classes,
                     base_features=base_features, d_state=d_state).to(device)

    # ── Multi-GPU: DataParallel ───────────────────────────────────────────────
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        device_ids = list(range(n_gpus))
        print(f"[Train] DataParallel across GPUs {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        for i in device_ids:
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {free // 1024**2} MB free / {total // 1024**2} MB total")
    else:
        print("[Train] Single GPU / CPU — no DataParallel")

    # ── Loss, Optimiser, Scheduler, AMP scaler ────────────────────────────────
    criterion = CombinedLoss(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_dsc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, scaler, epoch=epoch)
        scheduler.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            metrics = evaluate_batch(model, val_loader, device, num_classes)
            mean_dsc = metrics.get("DSC", {}).get("mean_dsc", 0.0)
            print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} | "
                  f"val DSC(mean)={mean_dsc:.4f}")

            if mean_dsc > best_dsc:
                best_dsc = mean_dsc
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
                # Unwrap DataParallel before saving so inference can load plain MambaBTS
                state = (model.module.state_dict()
                         if isinstance(model, nn.DataParallel)
                         else model.state_dict())
                torch.save({
                    "epoch": epoch,
                    "model_state": state,
                    "optimizer_state": optimizer.state_dict(),
                    "best_dsc": best_dsc,
                }, ckpt_path)
                print(f"  ↳ New best DSC={best_dsc:.4f}. Saved → {ckpt_path}")
        else:
            print(f"[Epoch {epoch:03d}] loss={train_loss:.4f}")

    print(f"\nTraining complete. Best validation DSC: {best_dsc:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — INFERENCE
# ─────────────────────────────────────────────────────────────────────────────


def run_inference(
    checkpoint_path: str,
    modality_paths: dict,
    output_path: str = "prediction.nii.gz",
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    num_classes: int = 4,
    base_features: int = 32,
    d_state: int = 16,
    device_str: str = "auto",
) -> np.ndarray:
    """
    Load a trained MambaBTS checkpoint and predict the segmentation mask for
    a single subject.

    Args:
        checkpoint_path : path to .pth checkpoint saved during training.
        modality_paths  : dict {"t1": ..., "t1ce": ..., "t2": ..., "flair": ...}
        output_path     : where to write the NIfTI prediction.
    Returns:
        pred : integer label map (H, W, D) in original image space.
    """
    if device_str == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(device_str)

    model = MambaBTS(in_channels=4, num_classes=num_classes,
                     base_features=base_features, d_state=d_state)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    preprocessor = BraTSPreprocessor()
    volume, _ = preprocessor.preprocess_subject(modality_paths)
    volume_t = torch.from_numpy(volume).unsqueeze(0).to(device)  # (1,4,H,W,D)

    with torch.no_grad():
        logits = model(volume_t)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ── Save prediction as NIfTI ──────────────────────────────────────────────
    ref_img = nib.load(list(modality_paths.values())[0])
    pred_nii = nib.Nifti1Image(pred, affine=ref_img.affine,
                                header=ref_img.header)
    nib.save(pred_nii, output_path)
    print(f"Prediction saved to: {output_path}")
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SMOKE TEST  (runs without a real BraTS dataset)
# ─────────────────────────────────────────────────────────────────────────────


def smoke_test(device_str: str = "cpu"):
    """
    End-to-end verification using random tensors.  No dataset required.
    Validates: model forward pass, Dice loss, DSC metric, HD95 metric.
    """
    print("=" * 60)
    print("MambaBTS Smoke Test")
    print("=" * 60)

    device = torch.device(device_str)
    B, C, H, W, D = 1, 4, 64, 64, 64
    num_classes = 4

    # ── Model instantiation ───────────────────────────────────────────────────
    print("\n[1/4] Instantiating MambaBTS model...")
    model = MambaBTS(in_channels=C, num_classes=num_classes,
                     base_features=16, d_state=8).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("\n[2/4] Forward pass on random input...")
    x = torch.randn(B, C, H, W, D, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (B, num_classes, H, W, D), \
        f"Output shape mismatch: {logits.shape}"
    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}  ✓")

    # ── Loss computation ──────────────────────────────────────────────────────
    print("\n[3/4] Loss computation (Dice + CE)...")
    criterion = CombinedLoss(num_classes=num_classes)
    target = torch.randint(0, num_classes, (B, H, W, D), device=device)
    loss = criterion(logits, target)
    print(f"  Combined loss: {loss.item():.4f}  ✓")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluation metrics (DSC + HD95)...")
    pred_np = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    tgt_np = target.squeeze(0).cpu().numpy()

    dsc = dice_similarity_coefficient(pred_np, tgt_np, num_classes)
    hd95 = hausdorff_distance_95(pred_np, tgt_np, num_classes)

    print("  DSC results:")
    for k, v in dsc.items():
        print(f"    {k:12s}: {v:.4f}")
    print("  HD95 results (mm):")
    for k, v in hd95.items():
        val_str = f"{v:.2f}" if not np.isinf(v) else "inf"
        print(f"    {k:12s}: {val_str}")

    print("\n" + "=" * 60)
    print("Smoke test PASSED ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MambaBTS — Brain Tumor Segmentation with State Space Models"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── smoke-test ────────────────────────────────────────────────────────────
    p_test = subparsers.add_parser("smoke-test",
                                   help="Run a quick sanity check (no data needed)")
    p_test.add_argument("--device", default="cpu")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train MambaBTS on BraTS data")
    p_train.add_argument("--data-root",       required=True)
    p_train.add_argument("--epochs",          type=int,   default=100)
    p_train.add_argument("--batch-size",      type=int,   default=1)
    p_train.add_argument("--lr",              type=float, default=1e-4)
    p_train.add_argument("--patch-size",      type=int,   nargs=3,
                         default=[128, 128, 128], metavar=("H", "W", "D"))
    p_train.add_argument("--base-features",   type=int,   default=32)
    p_train.add_argument("--d-state",         type=int,   default=16)
    p_train.add_argument("--checkpoint-dir",  default="checkpoints")
    p_train.add_argument("--device",          default="auto")

    # ── infer ─────────────────────────────────────────────────────────────────
    p_infer = subparsers.add_parser("infer", help="Run inference on a single subject")
    p_infer.add_argument("--checkpoint",  required=True)
    p_infer.add_argument("--t1",         required=True)
    p_infer.add_argument("--t1ce",       required=True)
    p_infer.add_argument("--t2",         required=True)
    p_infer.add_argument("--flair",      required=True)
    p_infer.add_argument("--output",     default="prediction.nii.gz")
    p_infer.add_argument("--device",     default="auto")

    args = parser.parse_args()

    if args.command == "smoke-test":
        smoke_test(device_str=args.device)

    elif args.command == "train":
        train(
            data_root=args.data_root,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patch_size=tuple(args.patch_size),
            base_features=args.base_features,
            d_state=args.d_state,
            checkpoint_dir=args.checkpoint_dir,
            device_str=args.device,
        )

    elif args.command == "infer":
        run_inference(
            checkpoint_path=args.checkpoint,
            modality_paths={
                "t1": args.t1, "t1ce": args.t1ce,
                "t2": args.t2, "flair": args.flair,
            },
            output_path=args.output,
            device_str=args.device,
        )

    else:
        parser.print_help()
