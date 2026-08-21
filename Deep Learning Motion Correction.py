

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIG
# =============================================================================
MODE = ""  # "train", "infer", or "check"

DATA_ROOT = r""
OUT_DIR = r""

# Used only by MODE == "infer". This may contain either:
#   1. scan folders with stack4d_before.npy, or
#   2. nested raw DICOM scan folders.
INFER_ROOT = r""
INFER_OUT = os.path.join(OUT_DIR, "inference")
CHECKPOINT = os.path.join(OUT_DIR, "best_model_cascade3.pth")
AUTO_CONVERT_RAW = True
RAW_PREPARED_DIR = os.path.join(OUT_DIR, "prepared_input")

# Model and temporal grouping.
GROUP_SIZE = 8
GROUP_STRIDE_INFERENCE = 4
WORKING_RESOLUTION = 192
BASE_CHANNELS = 16
UNET_DEPTH = 3
CASCADE_STAGES = 3
DROPOUT = 0.10

# Training.
EPOCHS = 50
GROUPS_PER_SCAN = 8
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
VAL_FRACTION = 0.10
USE_AMP = True
SEED = 42

# Four simple losses.
LAMBDA_FIELD = 10.0
LAMBDA_IMAGE = 1.0
LAMBDA_TEMPORAL = 0.25
LAMBDA_SMOOTH = 0.01
MOTION_WEIGHT_FLOOR = 0.10
HUBER_BETA = 0.10

# Exeter MDR fields are stored as (dy, dx); Code predicts (dx, dy).
FIELD_SWAP_XY = True
MOTION_THRESHOLD_PX = 0.25

# Optional checkpoint. Leave empty for fresh.
RESUME_CHECKPOINT = ""

# Inference.
INFER_GROUP_BATCH = 1
WARP_CHUNK_FRAMES = 8
VIDEO_FPS = 10
# =============================================================================


@dataclass(frozen=True)
class ModelConfig:
    group_size: int = GROUP_SIZE
    working_resolution: int = WORKING_RESOLUTION
    base_channels: int = BASE_CHANNELS
    unet_depth: int = UNET_DEPTH
    cascade_stages: int = CASCADE_STAGES
    dropout: float = DROPOUT


def configure_logging(output_dir: str, filename: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("simple_groupwise_v13")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, filename), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


log = logging.getLogger("simple_groupwise_v13")
if not log.handlers:
    log.addHandler(logging.NullHandler())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robust_intensity_limits(stack_path: Path) -> tuple[float, float]:
    stack = np.load(stack_path, mmap_mode="r")
    frame_count = min(16, stack.shape[0])
    indices = np.linspace(0, stack.shape[0] - 1, frame_count).round().astype(int)
    sample = np.asarray(stack[indices], dtype=np.float32)
    low, high = np.percentile(sample, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(sample))
        high = float(np.nanmax(sample))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def discover_training_scans(root: str) -> list[dict]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    scans: list[dict] = []
    for directory in sorted(path for path in root_path.iterdir() if path.is_dir()):
        before = directory / "stack4d_before.npy"
        fields = directory / "fields.npy"
        after = directory / "stack4d_after.npy"
        if not before.is_file() or not fields.is_file() or not after.is_file():
            continue
        image = np.load(before, mmap_mode="r")
        field = np.load(fields, mmap_mode="r")
        corrected = np.load(after, mmap_mode="r")
        if image.ndim != 4:
            log.warning("Skipping %s: before shape %s", directory.name, image.shape)
            continue
        if corrected.shape != image.shape:
            log.warning(
                "Skipping %s: MDR after shape %s does not match before %s",
                directory.name,
                corrected.shape,
                image.shape,
            )
            continue
        frames, depth, height, width = image.shape
        frame_first = field.shape == (frames, depth, height, width, 2)
        legacy = field.shape == (depth, height, width, frames, 2)
        if not frame_first and not legacy:
            log.warning(
                "Skipping %s: fields shape %s does not match before %s",
                directory.name,
                field.shape,
                image.shape,
            )
            continue
        low, high = robust_intensity_limits(before)
        scans.append(
            {
                "name": directory.name,
                "directory": str(directory),
                "before": str(before),
                "after": str(after),
                "fields": str(fields),
                "shape": tuple(int(value) for value in image.shape),
                "field_layout": "frame_first" if frame_first else "legacy",
                "low": low,
                "high": high,
            }
        )
    if not scans:
        raise RuntimeError(f"No labelled scans found under {root_path}")
    return scans


def split_scans(scans: list[dict]) -> tuple[list[dict], list[dict]]:
    generator = np.random.default_rng(SEED)
    order = generator.permutation(len(scans))
    validation_count = max(1, int(round(len(scans) * VAL_FRACTION)))
    validation_ids = set(int(index) for index in order[:validation_count])
    train = [scan for index, scan in enumerate(scans) if index not in validation_ids]
    validation = [
        scan for index, scan in enumerate(scans) if index in validation_ids
    ]
    return train, validation


def resize_images(images: torch.Tensor, resolution: int) -> torch.Tensor:
    """Resize (...,Z,H,W) in-plane while preserving all leading dimensions."""
    leading = images.shape[:-3]
    depth, height, width = images.shape[-3:]
    if (height, width) == (resolution, resolution):
        return images
    flat = images.reshape(-1, 1, height, width)
    resized = F.interpolate(
        flat,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(*leading, depth, resolution, resolution)


def resize_fields(
    fields: torch.Tensor,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Resize pixel-unit fields shaped (...,2,Z,H,W), including magnitudes."""
    leading = fields.shape[:-4]
    channels, depth, height, width = fields.shape[-4:]
    if channels != 2:
        raise ValueError(f"Expected two field channels, got {fields.shape}")
    if (height, width) == (output_height, output_width):
        return fields
    flat = fields.reshape(-1, 1, height, width)
    resized = F.interpolate(
        flat,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    ).reshape(*leading, channels, depth, output_height, output_width)
    resized = resized.clone()
    resized[..., 0, :, :, :] *= output_width / width
    resized[..., 1, :, :, :] *= output_height / height
    return resized


def load_field_group(
    scan: dict,
    indices: np.ndarray,
    resolution: int,
) -> torch.Tensor:
    source = np.load(scan["fields"], mmap_mode="r")
    if scan["field_layout"] == "frame_first":
        values = np.asarray(source[indices], dtype=np.float32)
    else:
        values = np.asarray(source[..., indices, :], dtype=np.float32)
        values = np.transpose(values, (3, 0, 1, 2, 4))
    fields = torch.from_numpy(np.ascontiguousarray(values).copy()).permute(
        0, 4, 1, 2, 3
    )
    if FIELD_SWAP_XY:
        fields = fields[:, [1, 0]]
    return resize_fields(fields, resolution, resolution)


class GroupDataset(Dataset):
    """Samples short contiguous frame groups from each scan."""

    def __init__(
        self,
        scans: list[dict],
        group_size: int,
        groups_per_scan: int,
        resolution: int,
        training: bool,
    ) -> None:
        self.scans = scans
        self.group_size = int(group_size)
        self.groups_per_scan = int(groups_per_scan)
        self.resolution = int(resolution)
        self.training = bool(training)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.scans) * self.groups_per_scan

    def _start(self, scan_index: int, slot: int, frames: int) -> int:
        maximum = max(0, frames - self.group_size)
        if maximum == 0:
            return 0
        if not self.training:
            starts = np.linspace(0, maximum, self.groups_per_scan)
            return int(round(starts[slot]))
        generator = np.random.default_rng(
            SEED + self.epoch * 1_000_003 + scan_index * 101 + slot
        )
        return int(generator.integers(0, maximum + 1))

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scan_index = item % len(self.scans)
        slot = item // len(self.scans)
        scan = self.scans[scan_index]
        frames = scan["shape"][0]
        start = self._start(scan_index, slot, frames)
        indices = np.arange(start, start + self.group_size, dtype=np.int64)
        indices = np.clip(indices, 0, frames - 1)

        source = np.load(scan["before"], mmap_mode="r")
        images_np = np.asarray(source[indices], dtype=np.float32)
        images = resize_images(
            torch.from_numpy(np.ascontiguousarray(images_np).copy()),
            self.resolution,
        )
        images = (images - scan["low"]) / max(scan["high"] - scan["low"], 1e-6)
        images = images.clamp(0.0, 1.0)
        after_source = np.load(scan["after"], mmap_mode="r")
        after_np = np.asarray(after_source[indices], dtype=np.float32)
        after = resize_images(
            torch.from_numpy(np.ascontiguousarray(after_np).copy()),
            self.resolution,
        )
        after = (after - scan["low"]) / max(
            scan["high"] - scan["low"], 1e-6
        )
        after = after.clamp(0.0, 1.0)
        fields = load_field_group(scan, indices, self.resolution)
        return images, fields, after


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class RegistrationUNet3D(nn.Module):
    

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = [2] + [
            config.base_channels * (2**level)
            for level in range(config.unet_depth + 1)
        ]
        pool = (1, 2, 2)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for level in range(config.unet_depth):
            self.encoders.append(
                ConvBlock3D(
                    channels[level],
                    channels[level + 1],
                    config.dropout if level > 0 else 0.0,
                )
            )
            self.pools.append(nn.MaxPool3d(pool, pool))
        self.bottleneck = ConvBlock3D(
            channels[config.unet_depth],
            channels[config.unet_depth + 1],
            config.dropout,
        )
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for stage in range(config.unet_depth):
            input_channels = channels[config.unet_depth + 1 - stage]
            output_channels = channels[config.unet_depth - stage]
            self.upconvs.append(
                nn.ConvTranspose3d(
                    input_channels,
                    output_channels,
                    kernel_size=pool,
                    stride=pool,
                )
            )
            self.decoders.append(
                ConvBlock3D(
                    output_channels * 2,
                    output_channels,
                    config.dropout,
                )
            )
        self.field_head = nn.Conv3d(channels[1], 2, kernel_size=1)
        nn.init.normal_(self.field_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.field_head.bias)

    def forward(
        self, moving: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        features = torch.stack([moving, reference], dim=1)
        skips: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            features = encoder(features)
            skips.append(features)
            features = pool(features)
        features = self.bottleneck(features)
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            features = upconv(features)
            if features.shape[2:] != skip.shape[2:]:
                features = F.interpolate(
                    features,
                    size=skip.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            features = decoder(torch.cat([features, skip], dim=1))
        return self.field_head(features)


class SpatialTransformer2D(nn.Module):
    """Warp each Z slice using a pixel-unit in-plane field."""

    def forward(
        self, image: torch.Tensor, field: torch.Tensor
    ) -> torch.Tensor:
        batch, depth, height, width = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(
                -1.0, 1.0, height, dtype=image.dtype, device=image.device
            ),
            torch.linspace(
                -1.0, 1.0, width, dtype=image.dtype, device=image.device
            ),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        displacement = field.permute(0, 2, 3, 4, 1).reshape(
            batch * depth, height, width, 2
        )
        scale = displacement.new_tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
            ]
        ).view(1, 1, 1, 2)
        grid = base + displacement * scale
        warped = F.grid_sample(
            image.reshape(batch * depth, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped[:, 0].reshape(batch, depth, height, width)


class CascadedGroupwiseUNet(nn.Module):
    """Three residual U-Nets trained jointly against one group mean."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.group_size = config.group_size
        self.stages = nn.ModuleList(
            [RegistrationUNet3D(config) for _ in range(config.cascade_stages)]
        )
        self.transformer = SpatialTransformer2D()

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, group, depth, height, width = frames.shape
        reference = frames.mean(dim=1)
        original = frames.reshape(batch * group, depth, height, width)
        reference_batch = (
            reference[:, None]
            .expand(batch, group, depth, height, width)
            .reshape(batch * group, depth, height, width)
        )
        total_field = original.new_zeros(
            batch * group, 2, depth, height, width
        )
        current = original
        for stage in self.stages:
            residual = stage(current, reference_batch)
            total_field = total_field + residual
            current = self.transformer(original, total_field)
        return (
            total_field.reshape(batch, group, 2, depth, height, width),
            current.reshape(batch, group, depth, height, width),
        )


class V13Loss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        warped: torch.Tensor,
        target_after: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        residual = F.smooth_l1_loss(
            prediction,
            target,
            beta=HUBER_BETA,
            reduction="none",
        )
        # Normalize motion weighting independently for dx and dy. This follows
        # the successful 17-June formulation and prevents zero-field collapse.
        reduce_dimensions = (0, 1, 3, 4, 5)
        target_absolute = target.abs()
        channel_mean = target_absolute.mean(
            dim=reduce_dimensions, keepdim=True
        )
        weights = (
            target_absolute + MOTION_WEIGHT_FLOOR * channel_mean
        )
        per_channel = (
            (weights * residual).sum(dim=reduce_dimensions)
            / weights.sum(dim=reduce_dimensions).clamp_min(1e-8)
        )
        field = per_channel.sum()
        field_dx, field_dy = per_channel[0], per_channel[1]

        image = F.smooth_l1_loss(
            warped,
            target_after,
            beta=0.01,
        )

        if prediction.shape[1] > 1:
            temporal = F.smooth_l1_loss(
                prediction[:, 1:] - prediction[:, :-1],
                target[:, 1:] - target[:, :-1],
                beta=HUBER_BETA,
            )
        else:
            temporal = prediction.new_tensor(0.0)

        gradients = []
        if prediction.shape[-3] > 1:
            gradients.append(
                (prediction[..., 1:, :, :] - prediction[..., :-1, :, :])
                .square()
                .mean()
            )
        gradients.append(
            (prediction[..., 1:, :] - prediction[..., :-1, :]).square().mean()
        )
        gradients.append(
            (prediction[..., 1:] - prediction[..., :-1]).square().mean()
        )
        smooth = torch.stack(gradients).mean()

        total = (
            LAMBDA_FIELD * field
            + LAMBDA_IMAGE * image
            + LAMBDA_TEMPORAL * temporal
            + LAMBDA_SMOOTH * smooth
        )
        metrics = {
            "total": float(total.detach()),
            "field": float(field.detach()),
            "field_dx": float(field_dx.detach()),
            "field_dy": float(field_dy.detach()),
            "image": float(image.detach()),
            "temporal": float(temporal.detach()),
            "smooth": float(smooth.detach()),
        }
        return total, metrics


@torch.no_grad()
def field_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    error = (prediction - target).square().sum(dim=2).add(1e-8).sqrt()
    target_magnitude = target.square().sum(dim=2).add(1e-8).sqrt()
    moving = target_magnitude > MOTION_THRESHOLD_PX
    if moving.any():
        cosine = F.cosine_similarity(
            prediction, target, dim=2, eps=1e-4
        )[moving].mean()
        angle = torch.rad2deg(
            torch.acos(cosine.clamp(-1.0, 1.0))
        )
        epe = error[moving].mean()
    else:
        cosine = prediction.new_tensor(float("nan"))
        angle = prediction.new_tensor(float("nan"))
        epe = error.mean()
    dx_ratio = (
        prediction[:, :, 0].abs().mean()
        / target[:, :, 0].abs().mean().clamp_min(1e-6)
    )
    dy_ratio = (
        prediction[:, :, 1].abs().mean()
        / target[:, :, 1].abs().mean().clamp_min(1e-6)
    )
    magnitude_ratio = (
        prediction.square().sum(dim=2).sqrt().mean()
        / target_magnitude.mean().clamp_min(1e-6)
    )
    return {
        "epe": float(epe),
        "cosine": float(cosine),
        "angle": float(angle),
        "dx_ratio": float(dx_ratio),
        "dy_ratio": float(dy_ratio),
        "magnitude_ratio": float(magnitude_ratio),
    }


def average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {
        key: float(np.nanmean([item[key] for item in items]))
        for key in keys
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: V13Loss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    records: list[dict[str, float]] = []
    amp_enabled = USE_AMP and device.type == "cuda"
    for images, target, target_after in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_after = target_after.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=torch.float16,
            ):
                prediction, warped = model(images)
                loss, loss_metrics = criterion(
                    prediction, target, warped, target_after
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        records.append({**loss_metrics, **field_metrics(prediction, target)})
    return average_metrics(records)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation: dict[str, float],
    config: ModelConfig,
    train_scans: list[dict],
    validation_scans: list[dict],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "model_config": asdict(config),
            "train_scans": [scan["name"] for scan in train_scans],
            "validation_scans": [scan["name"] for scan in validation_scans],
        },
        path,
    )


def train_v13() -> None:
    global log
    log = configure_logging(OUT_DIR, "training_cascade3.log")
    seed_everything(SEED)
    device = choose_device()
    scans = discover_training_scans(DATA_ROOT)
    train_scans, validation_scans = split_scans(scans)
    config = ModelConfig()
    log.info("V13: one-stage three-U-Net groupwise cascade")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info(
        "Scans: %d train / %d validation",
        len(train_scans),
        len(validation_scans),
    )
    log.info("Model config: %s", json.dumps(asdict(config), indent=2))
    log.info(
        "Loss = %.2f*field + %.2f*image + %.2f*temporal + %.3f*smooth",
        LAMBDA_FIELD,
        LAMBDA_IMAGE,
        LAMBDA_TEMPORAL,
        LAMBDA_SMOOTH,
    )

    train_dataset = GroupDataset(
        train_scans,
        config.group_size,
        GROUPS_PER_SCAN,
        config.working_resolution,
        training=True,
    )
    validation_dataset = GroupDataset(
        validation_scans,
        config.group_size,
        max(2, GROUPS_PER_SCAN // 2),
        config.working_resolution,
        training=False,
    )
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_options
    )

    model = CascadedGroupwiseUNet(config).to(device)
    criterion = V13Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=USE_AMP and device.type == "cuda",
    )
    start_epoch = 1
    best_validation = math.inf

    if RESUME_CHECKPOINT:
        checkpoint = torch.load(
            RESUME_CHECKPOINT, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["validation"]["total"])
        log.info("Resumed from epoch %d", start_epoch - 1)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log.info("Model parameters: %s", f"{parameter_count:,}")
    history_path = Path(OUT_DIR) / "training_history_cascade3.jsonl"

    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.time()
        train_dataset.set_epoch(epoch)
        training = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        validation = run_epoch(
            model, validation_loader, criterion, device, None, scaler
        )
        scheduler.step()
        elapsed = time.time() - started
        learning_rate = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %03d/%03d [%ds] lr=%.2e "
            "train=%.4f val=%.4f field=%.4f image=%.4f "
            "temporal=%.4f smooth=%.4f "
            "EPE=%.3fpx cos=%.3f angle=%.1f dx=%.2f dy=%.2f mag=%.2f",
            epoch,
            EPOCHS,
            round(elapsed),
            learning_rate,
            training["total"],
            validation["total"],
            validation["field"],
            validation["image"],
            validation["temporal"],
            validation["smooth"],
            validation["epe"],
            validation["cosine"],
            validation["angle"],
            validation["dx_ratio"],
            validation["dy_ratio"],
            validation["magnitude_ratio"],
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "elapsed_seconds": elapsed,
                        "learning_rate": learning_rate,
                        "train": training,
                        "validation": validation,
                    }
                )
                + "\n"
            )
        save_checkpoint(
            str(Path(OUT_DIR) / "last_model_cascade3.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            validation,
            config,
            train_scans,
            validation_scans,
        )
        if validation["total"] < best_validation:
            best_validation = validation["total"]
            save_checkpoint(
                CHECKPOINT,
                model,
                optimizer,
                scheduler,
                epoch,
                validation,
                config,
                train_scans,
                validation_scans,
            )
            log.info("Saved new best checkpoint: val=%.4f", best_validation)


def inference_starts(frames: int, group_size: int, stride: int) -> list[int]:
    if frames <= group_size:
        return [0]
    starts = list(range(0, frames - group_size + 1, stride))
    final = frames - group_size
    if starts[-1] != final:
        starts.append(final)
    return starts


@torch.no_grad()
def predict_full_series(
    model: CascadedGroupwiseUNet,
    stack: np.ndarray,
    low: float,
    high: float,
    config: ModelConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames, depth, height, width = stack.shape
    accumulation = torch.zeros(
        frames, 2, depth, height, width, dtype=torch.float32
    )
    counts = torch.zeros(frames, 1, 1, 1, 1, dtype=torch.float32)
    reference_accumulation = torch.zeros(
        frames,
        depth,
        config.working_resolution,
        config.working_resolution,
        dtype=torch.float32,
    )
    starts = inference_starts(
        frames, config.group_size, GROUP_STRIDE_INFERENCE
    )
    amp_enabled = USE_AMP and device.type == "cuda"

    for start in starts:
        indices = np.arange(start, start + config.group_size)
        valid = indices < frames
        clipped = np.clip(indices, 0, frames - 1)
        group_np = np.asarray(stack[clipped], dtype=np.float32)
        group = resize_images(
            torch.from_numpy(np.ascontiguousarray(group_np).copy()),
            config.working_resolution,
        )
        group = ((group - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
        group_reference = (
            group.mean(dim=0) * max(high - low, 1e-6) + low
        )
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            prediction, _ = model(group.unsqueeze(0).to(device))
            prediction = prediction[0].float().cpu()
        prediction = resize_fields(prediction, height, width)
        for offset, frame_index in enumerate(indices[valid]):
            accumulation[frame_index] += prediction[offset]
            reference_accumulation[frame_index] += group_reference
            counts[frame_index] += 1.0
    safe_counts = counts.clamp_min(1.0)
    fields = accumulation / safe_counts
    references = reference_accumulation / safe_counts[:, 0]
    return fields, references


def warp_frames(
    images: torch.Tensor,
    fields: torch.Tensor,
) -> torch.Tensor:
    """Warp (T,Z,H,W) images with pixel-unit (T,2,Z,H,W) fields."""
    frames, depth, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=images.device),
        torch.linspace(-1.0, 1.0, width, device=images.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)
    outputs = []
    for start in range(0, frames, WARP_CHUNK_FRAMES):
        stop = min(frames, start + WARP_CHUNK_FRAMES)
        image_chunk = images[start:stop].reshape(-1, 1, height, width)
        field_chunk = fields[start:stop].permute(0, 2, 3, 4, 1).reshape(
            -1, height, width, 2
        )
        normalized = field_chunk.clone()
        normalized[..., 0] *= 2.0 / max(width - 1, 1)
        normalized[..., 1] *= 2.0 / max(height - 1, 1)
        grid = base.unsqueeze(0) + normalized
        warped = F.grid_sample(
            image_chunk,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        outputs.append(warped[:, 0].reshape(stop - start, depth, height, width))
    return torch.cat(outputs, dim=0)


def save_preview(array: np.ndarray, output: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        log.warning("imageio unavailable; skipping %s", output)
        return
    middle = np.asarray(array[:, array.shape[1] // 2], dtype=np.float32)
    low, high = np.percentile(middle, [1.0, 99.5])
    scaled = ((middle - low) / max(high - low, 1e-6) * 255.0).clip(
        0, 255
    ).astype(np.uint8)
    rgb = np.repeat(scaled[..., None], 3, axis=-1)
    imageio.mimwrite(output, rgb, fps=fps, macro_block_size=1)


def copy_optional_file(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def link_or_copy(source: Path, destination: Path) -> None:
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def save_qc_outputs(
    before: np.ndarray,
    after: np.ndarray,
    output_dir: Path,
) -> None:
    
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        middle_z = before.shape[1] // 2
        frame_indices = np.linspace(
            0, before.shape[0] - 1, min(6, before.shape[0])
        ).round().astype(int)
        selected_before = np.asarray(
            before[frame_indices, middle_z], dtype=np.float32
        )
        selected_after = np.asarray(
            after[frame_indices, middle_z], dtype=np.float32
        )
        low, high = np.percentile(selected_before, [1.0, 99.5])

        figure, axes = plt.subplots(
            3, len(frame_indices), figsize=(2.3 * len(frame_indices), 6.5)
        )
        if len(frame_indices) == 1:
            axes = axes.reshape(3, 1)
        difference_limit = float(
            np.percentile(
                np.abs(selected_after - selected_before), 99.0
            )
        )
        difference_limit = max(difference_limit, 1e-6)
        for column, frame_index in enumerate(frame_indices):
            axes[0, column].imshow(
                selected_before[column], cmap="gray", vmin=low, vmax=high
            )
            axes[1, column].imshow(
                selected_after[column], cmap="gray", vmin=low, vmax=high
            )
            axes[2, column].imshow(
                np.abs(selected_after[column] - selected_before[column]),
                cmap="magma",
                vmin=0.0,
                vmax=difference_limit,
            )
            axes[0, column].set_title(f"t={frame_index}")
            for row in range(3):
                axes[row, column].axis("off")
        axes[0, 0].set_ylabel("Before")
        axes[1, 0].set_ylabel("After")
        axes[2, 0].set_ylabel("|After-Before|")
        figure.tight_layout()
        figure.savefig(
            output_dir / "contact_before_after_diff.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)

        row = before.shape[2] // 2
        before_cut = np.asarray(before[:, middle_z, row], dtype=np.float32)
        after_cut = np.asarray(after[:, middle_z, row], dtype=np.float32)

        def save_timecut(values: np.ndarray, filename: str, title: str) -> None:
            fig, axis = plt.subplots(figsize=(10, 5))
            axis.imshow(
                values,
                cmap="gray",
                aspect="auto",
                vmin=low,
                vmax=high,
            )
            axis.set_title(title)
            axis.set_xlabel("W position")
            axis.set_ylabel("Time frame")
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

        save_timecut(before_cut, "timecut_before.png", "Before correction")
        save_timecut(after_cut, "timecut_after.png", "After correction")
        figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].imshow(
            before_cut, cmap="gray", aspect="auto", vmin=low, vmax=high
        )
        axes[1].imshow(
            after_cut, cmap="gray", aspect="auto", vmin=low, vmax=high
        )
        axes[0].set_title("Before correction")
        axes[1].set_title("After correction")
        axes[1].set_xlabel("W position")
        for axis in axes:
            axis.set_ylabel("Time frame")
        figure.tight_layout()
        figure.savefig(
            output_dir / "timecut_compare.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)
        log.info("Saved V6-compatible contact-sheet and timecut PNGs.")
    except Exception as error:
        log.warning("PNG QC generation failed (non-fatal): %s", error)


def discover_inference_scans(root: str) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    # A single prepared scan may be supplied directly.
    if (root_path / "stack4d_before.npy").is_file():
        log.info("Using prepared inference scan: %s", root_path)
        return [root_path]

    # Preserve the existing multi-scan stack4d_before.npy workflow.
    scans = [
        directory
        for directory in sorted(root_path.iterdir())
        if directory.is_dir()
        and (directory / "stack4d_before.npy").is_file()
    ]
    if scans:
        log.info(
            "Using %d prepared inference scan(s) from %s",
            len(scans),
            root_path,
        )
        return scans

    if not AUTO_CONVERT_RAW:
        raise RuntimeError(
            f"No stack4d_before.npy scans found under {root_path}, and "
            "AUTO_CONVERT_RAW is disabled."
        )

   
    try:
        from dicom_to_npy import convert_batch
    except ImportError as error:
        raise ImportError(
            "Raw conversion requires dicom_to_npy.py and mri_reader.py beside "
            "simple_groupwise_v13.py."
        ) from error

    prepared_root = Path(RAW_PREPARED_DIR)
    prepared_root.mkdir(parents=True, exist_ok=True)
    log.info(
        "No prepared stacks found. Converting raw scans from %s to %s",
        root_path,
        prepared_root,
    )
    convert_batch(str(root_path), str(prepared_root))

    scans = [
        directory
        for directory in sorted(prepared_root.iterdir())
        if directory.is_dir()
        and (directory / "stack4d_before.npy").is_file()
    ]
    if not scans:
        raise RuntimeError(
            f"Raw conversion produced no stack4d_before.npy scans in "
            f"{prepared_root}. Check inference_cascade3.log and the terminal "
            "conversion messages."
        )
    log.info(
        "Prepared %d raw scan(s); starting V13 inference.",
        len(scans),
    )
    return scans


def infer_v13() -> None:
    global log
    log = configure_logging(OUT_DIR, "inference_cascade3.log")
    device = choose_device()
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    config = ModelConfig(**checkpoint["model_config"])
    model = CascadedGroupwiseUNet(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info(
        "Loaded V13 epoch %d on %s",
        int(checkpoint["epoch"]),
        device,
    )
    os.makedirs(INFER_OUT, exist_ok=True)

    for scan_directory in discover_inference_scans(INFER_ROOT):
        before_path = scan_directory / "stack4d_before.npy"
        source = np.load(before_path, mmap_mode="r")
        if source.ndim != 4:
            log.warning("Skipping %s: shape %s", scan_directory.name, source.shape)
            continue
        low, high = robust_intensity_limits(before_path)
        fields, reference_working = predict_full_series(
            model, source, low, high, config, device
        )
        before = torch.from_numpy(
            np.ascontiguousarray(np.asarray(source, dtype=np.float32)).copy()
        )
        corrected = warp_frames(before.to(device), fields.to(device)).cpu()

        output = Path(INFER_OUT) / scan_directory.name
        output.mkdir(parents=True, exist_ok=True)
        before_np = before.numpy()
        corrected_np = corrected.numpy()
        field_disk = fields.permute(2, 3, 4, 0, 1).numpy()
        corrected_path = output / "stack4d_after.npy"
        link_or_copy(before_path, output / "stack4d_before.npy")
        np.save(corrected_path, corrected_np)
        link_or_copy(corrected_path, output / "predicted_warped.npy")
        np.save(output / "predicted_fields.npy", field_disk)
        np.save(
            output / "predicted_reference_working.npy",
            reference_working.numpy(),
        )
        np.save(
            output / "predicted_trajectory.npy",
            fields.mean(dim=(2, 3, 4)).numpy(),
        )
        save_preview(before_np, output / "preview_before.mp4", VIDEO_FPS)
        save_preview(corrected_np, output / "preview_after.mp4", VIDEO_FPS)
        save_preview(
            reference_working.numpy(),
            output / "preview_reference.mp4",
            VIDEO_FPS,
        )

        fit_source = scan_directory / "stack4d_fit.npy"
        if copy_optional_file(fit_source, output / "stack4d_fit.npy"):
            fit = np.load(fit_source, mmap_mode="r")
            if fit.shape == source.shape:
                save_preview(
                    np.asarray(fit),
                    output / "preview_fit.mp4",
                    VIDEO_FPS,
                )
            else:
                log.warning(
                    "Skipping fit preview for %s: fit shape %s != before %s",
                    scan_directory.name,
                    fit.shape,
                    source.shape,
                )
        copy_optional_file(
            scan_directory / "crop_bounds.json",
            output / "crop_bounds.json",
        )
        copy_optional_file(
            scan_directory / "kidney_mask.npy",
            output / "kidney_mask.npy",
        )
        save_qc_outputs(before_np, corrected_np, output)
        log.info(
            "Saved %s: before/after=%s fields=%s plus V6-compatible QC",
            output,
            before_np.shape,
            field_disk.shape,
        )


def check_v13() -> None:
    global log
    log = configure_logging(OUT_DIR, "check_cascade3.log")
    device = choose_device()
    scans = discover_training_scans(DATA_ROOT)
    config = ModelConfig()
    dataset = GroupDataset(
        scans[:1],
        config.group_size,
        1,
        config.working_resolution,
        training=False,
    )
    images, target, target_after = dataset[0]
    model = CascadedGroupwiseUNet(config).to(device)
    criterion = V13Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    check_batch = min(BATCH_SIZE, 2)
    images_batch = images.unsqueeze(0).repeat(check_batch, 1, 1, 1, 1)
    target_batch = target.unsqueeze(0).repeat(check_batch, 1, 1, 1, 1, 1)
    after_batch = target_after.unsqueeze(0).repeat(
        check_batch, 1, 1, 1, 1
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=USE_AMP and device.type == "cuda",
        dtype=torch.float16,
    ):
        prediction, warped = model(images_batch.to(device))
        loss, metrics = criterion(
            prediction,
            target_batch.to(device),
            warped,
            after_batch.to(device),
        )
    loss.backward()
    optimizer.step()
    log.info("V13 check passed on %s", device)
    log.info("Input: %s", tuple(images_batch.shape))
    log.info("Target/output: %s", tuple(prediction.shape))
    log.info("Loss: %.4f (%s)", float(loss), metrics)


if __name__ == "__main__":
    if MODE == "train":
        train_v13()
    elif MODE == "infer":
        infer_v13()
    elif MODE == "check":
        check_v13()
    else:
        raise ValueError(f"MODE must be train, infer, or check; got {MODE!r}")
