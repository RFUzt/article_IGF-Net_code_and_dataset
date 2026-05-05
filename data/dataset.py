# data/dataset.py
"""
Dataset & Augmentation:
    - Augmentations are performed using Albumentations and registered via AUGMENT_REGISTRY.
    - NPYSegDataset: adapts multi-band .npy images in (C, H, W) format; supports "rgb" (3-channel selection) or "all" (full channel input);
    - Automatically matches mask suffixes (.png/.tif/.npy/.jpg/.jpeg/.bmp), reads and binarizes the mask.
    - Training/validation/testing return (image, mask, stem); inference (mask_dir=None) returns (image, stem).
"""
import os
import cv2
import glob
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from utils.registry import DATASET_REGISTRY, AUGMENT_REGISTRY


def _read_mask(mask_path):
    ext = os.path.splitext(mask_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        max_val = m.max()
        min_val = m.min()

        if max_val <= 1:

            return (m > 0).astype(np.uint8)

        return (m > 127).astype(np.uint8)

    elif ext == ".npy":
        m = np.load(mask_path)

        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]

        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]

        return (m > 0.5).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported mask type: {mask_path}")


def _percent_norm(x):

    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)

    lo = float(lo)
    hi = float(hi)

    if np.isnan(lo) or np.isnan(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo)/(hi-lo), a_min=0, a_max=1).astype(np.float32)


@AUGMENT_REGISTRY.register("basic_train")
def build_train_aug(img_size):
    ops = []
    if img_size is not None:
        ops.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST))

    ops += [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(transpose_mask=True)
    ]

    return A.Compose(ops)


@AUGMENT_REGISTRY.register("basic_val")
def build_val_aug(img_size):

    ops = []
    if img_size is not None:
        ops.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST))

    ops += [ToTensorV2(transpose_mask=True)]

    return A.Compose(ops)


@DATASET_REGISTRY.register("npyseg")
class NPYSegDataset(Dataset):

    def __init__(
            self,
            img_dir,
            mask_dir,
            mode,
            rgb_idx=(7,10,13),
            img_size=None,
            augment_name="basic_train",
            is_train=True
    ):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.rgb_idx = tuple(rgb_idx)

        self.files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy found in {img_dir}")

        aug_builder = AUGMENT_REGISTRY.get(augment_name)
        self.aug = aug_builder(img_size)

        if mode not in ["all", "rgb"]:
            raise ValueError("mode must be 'all' or 'rgb'")

    def __len__(self):
        return len(self.files)

    def _select(self, arr):
        """按 mode 选择通道。"""
        if self.mode == "rgb":
            r,g,b = self.rgb_idx
            return arr[[r,g,b], :, :]
        return arr

    def __getitem__(self, idx):
        img_path = self.files[idx]
        arr = np.load(img_path)

        if arr.ndim != 3 or (arr.shape[0] not in (14, 5)):
            raise ValueError(f"{img_path} shape must be (14,H,W) or (5,H,W), got {arr.shape}")

        arr = arr.astype(np.float32)
        arr = self._select(arr)

        for i in range(arr.shape[0]):
            arr[i] = _percent_norm(arr[i])

        sample = {"image": np.transpose(arr, axes=(1,2,0))}
        stem = os.path.splitext(os.path.basename(img_path))[0]

        if self.mask_dir is not None:
            mask_path = None

            for suf in [".png",".tif",".npy",".jpg",".jpeg",".bmp"]:
                cand = os.path.join(self.mask_dir, stem + suf)
                if os.path.exists(cand):
                    mask_path = cand
                    break

            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {img_path} in {self.mask_dir}")

            m = _read_mask(mask_path)
            sample["mask"] = m

            out = self.aug(image=sample["image"], mask=sample["mask"])
            img_t  = out["image"].float()
            mask_t = out["mask"].long()
            return img_t, mask_t, stem
        else:
            out = self.aug(image=sample["image"])
            img_t = out["image"].float()
            return img_t, stem
