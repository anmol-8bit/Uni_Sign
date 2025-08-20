# datasets_augmentations.py
"""
Data Augmentation Module for Sign Language Translation
Implements temporal and spatial augmentations specifically designed for pose keypoint data
in sign language recognition and translation tasks, with epoch-progress scheduling.
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Dict, Optional
import torch


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


class SignLanguageAugmentation:
    """
    Main augmentation class for sign language pose data.
    """

    def __init__(self,
                 temporal_aug_prob: float = 0.7,
                 spatial_aug_prob: float = 0.8,
                 speed_range: Tuple[float, float] = (0.8, 1.2),
                 translation_range: float = 0.1,
                 rotation_range: float = 15.0,
                 noise_std: float = 0.01,
                 dropout_prob: float = 0.1):
        self.temporal_aug_prob = temporal_aug_prob
        self.spatial_aug_prob = spatial_aug_prob
        self.speed_range = speed_range
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def apply_augmentations(self,
                            indices: np.ndarray,
                            skeletons: List,
                            confs: List,
                            duration: int) -> Tuple[np.ndarray, List, List]:
        # Temporal
        if random.random() < self.temporal_aug_prob:
            indices = self._apply_temporal_augmentations(indices, duration)
        # Spatial
        if random.random() < self.spatial_aug_prob:
            skeletons, confs = self._apply_spatial_augmentations(skeletons, confs)
        return indices, skeletons, confs

    # ---------- temporal ----------
    def _apply_temporal_augmentations(self, indices: np.ndarray, duration: int) -> np.ndarray:
        if random.random() < 0.6:
            indices = self._apply_speed_variation(indices, duration)
        if random.random() < 0.4:
            indices = self._apply_temporal_jittering(indices, duration)
        if random.random() < 0.3:
            indices = self._apply_temporal_dropout(indices)
        return indices

    def _apply_speed_variation(self, indices: np.ndarray, duration: int) -> np.ndarray:
        speed_factor = np.random.uniform(self.speed_range[0], self.speed_range[1])
        if speed_factor < 1.0:
            n_frames = max(int(len(indices) * speed_factor), 1)
            step = len(indices) / n_frames
            new_indices = [indices[int(i * step)] for i in range(n_frames)]
            return np.array(new_indices)
        elif speed_factor > 1.0:
            n_frames = min(int(len(indices) * speed_factor), duration)
            if n_frames > len(indices):
                repeat_factor = n_frames / len(indices)
                new_indices = []
                for idx in indices:
                    repeats = max(1, int(repeat_factor))
                    new_indices.extend([idx] * repeats)
                return np.array(new_indices[:n_frames])
        return indices

    def _apply_temporal_jittering(self, indices: np.ndarray, duration: int) -> np.ndarray:
        jitter_range = max(1, int(0.05 * len(indices)))
        jitter = np.random.randint(-jitter_range, jitter_range + 1)
        return np.clip(indices + jitter, 0, duration - 1)

    def _apply_temporal_dropout(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) < 4:
            return indices
        dropout_count = max(1, int(self.dropout_prob * len(indices)))
        keep_indices = np.random.choice(len(indices), size=len(indices) - dropout_count, replace=False)
        keep_indices = np.sort(keep_indices)
        return indices[keep_indices]

    # ---------- spatial ----------
    def _apply_spatial_augmentations(self, skeletons: List, confs: List) -> Tuple[List, List]:
        skeletons = copy.deepcopy(skeletons)
        confs = copy.deepcopy(confs)
        skeleton_array = np.array(skeletons)  # (T, 1, 133, 2)
        conf_array = np.array(confs)          # (T, 1, 133)

        if random.random() < 0.7:
            skeleton_array = self._apply_translation(skeleton_array)
        if random.random() < 0.5:
            skeleton_array = self._apply_rotation(skeleton_array)
        if random.random() < 0.4:
            skeleton_array = self._apply_scaling(skeleton_array)
        if random.random() < 0.5:
            skeleton_array = self._apply_noise_injection(skeleton_array, conf_array)
        if random.random() < 0.3:
            skeleton_array, conf_array = self._apply_keypoint_dropout(skeleton_array, conf_array)

        return skeleton_array.tolist(), conf_array.tolist()

    def _apply_translation(self, skeleton_array: np.ndarray) -> np.ndarray:
        valid_points = skeleton_array[skeleton_array[..., 0] != 0]
        if len(valid_points) == 0:
            return skeleton_array
        x_range = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
        y_range = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
        tx = np.random.uniform(-self.translation_range, self.translation_range) * x_range
        ty = np.random.uniform(-self.translation_range, self.translation_range) * y_range
        mask = skeleton_array[..., 0] != 0
        skeleton_array[mask, 0] += tx
        skeleton_array[mask, 1] += ty
        return skeleton_array

    def _apply_rotation(self, skeleton_array: np.ndarray) -> np.ndarray:
        T, _, N, _ = skeleton_array.shape
        for t in range(T):
            frame = skeleton_array[t, 0]
            torso_indices = [0, 1, 2, 5, 6]
            torso_pts = [frame[idx] for idx in torso_indices if idx < N and frame[idx, 0] != 0]
            if not torso_pts:
                continue
            center = np.mean(torso_pts, axis=0)
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            a = np.radians(angle)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            for i in range(N):
                if frame[i, 0] != 0:
                    p = frame[i] - center
                    skeleton_array[t, 0, i] = rot @ p + center
        return skeleton_array

    def _apply_scaling(self, skeleton_array: np.ndarray) -> np.ndarray:
        scale_factor = np.random.uniform(0.9, 1.1)
        valid_points = skeleton_array[skeleton_array[..., 0] != 0]
        if len(valid_points) == 0:
            return skeleton_array
        center = np.mean(valid_points, axis=0)
        mask = skeleton_array[..., 0] != 0
        skeleton_array[mask] = (skeleton_array[mask] - center) * scale_factor + center
        return skeleton_array

    def _apply_noise_injection(self, skeleton_array: np.ndarray, conf_array: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, skeleton_array.shape)
        conf_threshold = 0.5
        noise_scale = np.where(conf_array[..., None] < conf_threshold, 2.0, 1.0)
        noise *= noise_scale
        mask = skeleton_array[..., 0] != 0
        skeleton_array[mask] += noise[mask]
        return skeleton_array

    def _apply_keypoint_dropout(self, skeleton_array: np.ndarray, conf_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, _, N, _ = skeleton_array.shape
        for t in range(T):
            for i in range(N):
                base = self.dropout_prob
                p = base * 2.0 if conf_array[t, 0, i] < 0.3 else base
                if random.random() < p:
                    skeleton_array[t, 0, i] = [0, 0]
                    conf_array[t, 0, i] = 0.0
        return skeleton_array, conf_array


def create_augmentation_pipeline(phase: str, progress: float = 1.0) -> Optional[SignLanguageAugmentation]:
    """
    Schedule augmentation strength with training progress (0..1).
    We keep val/test clean.
    """
    if phase != 'train':
        return None

    # Probabilities: ramp from mild to strong
    temporal_aug_prob = _lerp(0.30, 0.70, progress)
    spatial_aug_prob  = _lerp(0.20, 0.80, progress)

    # Magnitudes widen with progress
    # e.g., speed range expands both <1 (skip) and >1 (repeat)
    speed_lo = _lerp(0.95, 0.80, progress)
    speed_hi = _lerp(1.05, 1.20, progress)

    translation_range = _lerp(0.02, 0.12, progress)
    rotation_range    = _lerp(5.0, 20.0, progress)
    noise_std         = _lerp(0.002, 0.02, progress)
    dropout_prob      = _lerp(0.02, 0.12, progress)

    return SignLanguageAugmentation(
        temporal_aug_prob=temporal_aug_prob,
        spatial_aug_prob=spatial_aug_prob,
        speed_range=(speed_lo, speed_hi),
        translation_range=translation_range,
        rotation_range=rotation_range,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
    )


def apply_pose_augmentations(indices: np.ndarray,
                             skeletons: List,
                             confs: List,
                             duration: int,
                             phase: str,
                             progress: float = 1.0) -> Tuple[np.ndarray, List, List]:
    aug = create_augmentation_pipeline(phase, progress)
    if aug is None:
        return indices, skeletons, confs
    return aug.apply_augmentations(indices, skeletons, confs, duration)


# Optional quick test
if __name__ == "__main__":
    # dummy check
    duration = 100
    max_length = 50
    idx = np.array(sorted(random.sample(range(duration), k=max_length)))
    skeletons = [[[np.random.rand(133, 2)] for _ in range(1)] for _ in range(len(idx))]
    confs = [[[np.random.rand(133)] for _ in range(1)] for _ in range(len(idx))]
    out = apply_pose_augmentations(idx, skeletons, confs, duration, 'train', progress=0.5)
    print("OK", len(out[1]))
