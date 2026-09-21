from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


# Seven demographic/clinical variables plus 62 quantitative OCTA variables.
# Original feature index 7 (acquisition-device version) is excluded.
FULL_NO_DEVICE_INDICES = list(range(0, 7)) + list(range(8, 70))


def build_transform(training: bool, probability: float, mean: float, std: float):
    operations = []
    if training:
        operations.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=probability,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=probability),
                A.CLAHE(p=probability),
                A.VerticalFlip(p=probability),
                A.HorizontalFlip(p=probability),
                A.AdvancedBlur(p=probability),
            ]
        )
    operations.extend([A.Normalize(mean=(mean,), std=(std,)), ToTensorV2()])
    return A.Compose(operations)


def _to_float(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"", "nan", "none"}:
        return 0.0
    if normalized in {"true", "yes"}:
        return 1.0
    if normalized in {"false", "no"}:
        return 0.0
    return float(normalized)


def load_split(split_file: Path, image_root: Path):
    image_dirs, labels, tabular_rows = [], [], []
    for line_number, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 72:
            raise ValueError(
                f"Expected path, label, and 70 tabular values at "
                f"{split_file}:{line_number}; found {len(fields)} columns"
            )
        image_path = Path(fields[0])
        if not image_path.is_absolute():
            image_path = image_root / image_path
        image_dirs.append(image_path)
        labels.append(int(float(fields[1])))
        original_tabular = [_to_float(value) for value in fields[2:]]
        tabular_rows.append([original_tabular[index] for index in FULL_NO_DEVICE_INDICES])
    if not image_dirs:
        raise ValueError(f"No samples found in {split_file}")
    return image_dirs, labels, tabular_rows


class OCTATabularDataset(Dataset):
    LAYER_SPECS = (
        ("svc", "_浅层血管复合体.png", 0.485, 0.229),
        ("dvc", "_深层血管复合体.png", 0.456, 0.224),
        ("cc", "_脉络膜毛细血管层.png", 0.406, 0.225),
    )

    def __init__(
        self,
        image_dirs,
        labels,
        tabular_rows,
        training: bool,
        vmin=None,
        value_range=None,
        augmentation_probability: float = 0.3,
    ):
        self.image_dirs = [Path(path) for path in image_dirs]
        self.labels = list(labels)
        self.tabular = np.asarray(tabular_rows, dtype=np.float32)
        self.training = training
        self.augmentation_probability = augmentation_probability

        if self.tabular.shape[1] != 69:
            raise ValueError(f"Expected 69 tabular features, got {self.tabular.shape[1]}")
        if vmin is None or value_range is None:
            normalization_mask = np.zeros(69, dtype=bool)
            normalization_mask[1] = True
            normalization_mask[7:] = True
            observed_min = self.tabular.min(axis=0)
            observed_max = self.tabular.max(axis=0)
            observed_range = np.maximum(observed_max - observed_min, 1e-8)
            vmin = np.where(normalization_mask, observed_min, 0.0)
            value_range = np.where(normalization_mask, observed_range, 1.0)
        self.vmin = torch.tensor(vmin, dtype=torch.float32)
        self.value_range = torch.tensor(value_range, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _find_layer(directory: Path, suffix: str) -> Path:
        matches = sorted(directory.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '*{suffix}' in {directory}, found {len(matches)}"
            )
        return matches[0]

    def __getitem__(self, index):
        directory = self.image_dirs[index]
        layer_tensors = []
        for _, suffix, mean, std in self.LAYER_SPECS:
            layer_path = self._find_layer(directory, suffix)
            with Image.open(layer_path) as image:
                array = np.asarray(image)
            transform = build_transform(
                self.training,
                probability=self.augmentation_probability,
                mean=mean,
                std=std,
            )
            layer_tensors.append(transform(image=array)["image"].float())

        tabular = torch.tensor(self.tabular[index], dtype=torch.float32)
        tabular = (tabular - self.vmin) / self.value_range
        return (*layer_tensors, tabular, int(self.labels[index]))
