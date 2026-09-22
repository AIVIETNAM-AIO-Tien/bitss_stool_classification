"""
Data pipeline cho bài toán phân loại BITSS Type 1-7.

Thiết kế theo đúng tinh thần đã thống nhất trong kế hoạch (Giai đoạn 2 & 4):
    - split_mode="image_level":         dùng cấu trúc train/valid/test có sẵn
                                         (Setup A / baseline).
    - split_mode="pseudo_patient_level": data tạm KHÔNG có patient_id thật ->
                                         giả lập pseudo-patient bằng cách gộp
                                         N ảnh liên tiếp cùng lớp thành 1
                                         "bệnh nhân ảo", rồi chia theo pseudo-ID
                                         để KIỂM THỬ logic patient-level split.
                                         KHÔNG dùng kết quả này để kết luận
                                         khoa học về generalization thật.
    - split_mode="patient_level":       dùng khi có metadata thật (CSV có cột
                                         patient_id) từ dữ liệu lâm sàng.

Khi chuyển sang dữ liệu thật, chỉ cần trỏ `patient_metadata_csv` trong
configs/data_paths.yaml và đổi split_mode -> "patient_level"; phần code
training/evaluate/explain phía sau không cần sửa.
"""
import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Sample:
    image_path: str
    label: int
    group_id: str  # patient_id thật, pseudo_patient_id, hoặc chính image_path (image_level)


def _list_image_folder_samples(split_dir: str, class_names: list[str]) -> list[Sample]:
    """Đọc cấu trúc ImageFolder chuẩn: split_dir/Type_i/*.jpg
    group_id mặc định = image_path (mỗi ảnh là 1 nhóm độc lập) cho image_level.
    """
    samples = []
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục {split_dir}. "
            f"Kiểm tra lại data.root_dir trong config và cấu trúc "
            f"data/train|valid|test/Type_1..7/*.jpg"
        )
    for label_idx, class_name in enumerate(class_names):
        class_dir = split_path / class_name
        if not class_dir.exists():
            warnings.warn(f"Thiếu thư mục lớp {class_dir} — bỏ qua lớp này trong split {split_dir}")
            continue
        for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.png")):
            samples.append(Sample(image_path=str(img_path), label=label_idx, group_id=str(img_path)))
    return samples


def _assign_pseudo_patient_ids(samples: list[Sample], images_per_group: int, seed: int) -> list[Sample]:
    """Giả lập pseudo-patient: gộp N ảnh liên tiếp (đã shuffle theo seed cố định,
    trong CÙNG một lớp) thành 1 group_id. Mục đích DUY NHẤT là kiểm thử logic
    GroupShuffleSplit không rò rỉ nhóm giữa train/test — KHÔNG phản ánh sinh học
    thật (ảnh trong data tạm không thật sự thuộc cùng 1 đứa trẻ).
    """
    rng = np.random.RandomState(seed)
    by_label: dict[int, list[Sample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    updated = []
    for label, group_samples in by_label.items():
        idx = np.arange(len(group_samples))
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            pseudo_id = f"pseudo_L{label}_{i // images_per_group}"
            s = group_samples[sample_idx]
            updated.append(Sample(image_path=s.image_path, label=s.label, group_id=pseudo_id))
    return updated


def _load_samples_from_metadata_csv(csv_path: str) -> list[Sample]:
    """Đọc metadata thật — dùng khi split_mode='patient_level'.
    CSV cần có tối thiểu 3 cột: image_path, patient_id, label
    (xem docs/data_contract.md để biết schema đầy đủ khuyến nghị).
    """
    df = pd.read_csv(csv_path)
    required_cols = {"image_path", "patient_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"metadata_csv thiếu cột bắt buộc: {missing}. "
            f"Xem docs/data_contract.md để biết schema yêu cầu."
        )
    return [
        Sample(image_path=row["image_path"], label=int(row["label"]), group_id=str(row["patient_id"]))
        for _, row in df.iterrows()
    ]


def get_transforms(image_size: int, split: str, aug_cfg: dict) -> transforms.Compose:
    if split == "train" and aug_cfg.get("train", {}):
        t_cfg = aug_cfg["train"]
        ops = [transforms.Resize((image_size, image_size))]
        if t_cfg.get("horizontal_flip"):
            ops.append(transforms.RandomHorizontalFlip())
        if t_cfg.get("rotation_degrees"):
            ops.append(transforms.RandomRotation(t_cfg["rotation_degrees"]))
        if t_cfg.get("color_jitter_brightness") or t_cfg.get("color_jitter_contrast"):
            ops.append(transforms.ColorJitter(
                brightness=t_cfg.get("color_jitter_brightness", 0),
                contrast=t_cfg.get("color_jitter_contrast", 0),
            ))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(ops)
    else:
        # valid/test: KHÔNG augmentation, chỉ resize + normalize
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class BitssStoolDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        return image, sample.label, sample.image_path


def _group_split(samples: list[Sample], test_size: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Chia samples theo group_id (patient_id hoặc pseudo_patient_id) sao cho
    không có group nào xuất hiện ở cả 2 tập — đây là logic cốt lõi để tránh
    data leakage đã nêu ở RQ1 (Tampu et al. 2022; 'How You Split Matters').
    """
    groups = [s.group_id for s in samples]
    labels = [s.label for s in samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X=np.zeros(len(samples)), y=labels, groups=groups))
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    # Kiểm tra bất biến quan trọng nhất: không rò rỉ group giữa 2 tập
    train_groups = {s.group_id for s in train_samples}
    test_groups = {s.group_id for s in test_samples}
    overlap = train_groups & test_groups
    assert not overlap, f"LỖI LOGIC SPLIT: {len(overlap)} group bị rò rỉ giữa train/test!"

    return train_samples, test_samples


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """Entry point chính. Trả về dict {'train': DataLoader, 'valid': ..., 'test': ...}
    theo đúng split_mode cấu hình trong configs/default.yaml.
    """
    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]
    class_names = data_cfg["class_names"]
    split_mode = data_cfg["split_mode"]
    seed = cfg.get("seed", 42)

    dataloaders = {}

    if split_mode == "image_level":
        # Dùng đúng cấu trúc train/valid/test có sẵn — không cần group split
        for split in ["train", "valid", "test"]:
            samples = _list_image_folder_samples(os.path.join(root_dir, split), class_names)
            transform = get_transforms(data_cfg["image_size"], split, cfg["augmentation"])
            ds = BitssStoolDataset(samples, transform)
            dataloaders[split] = DataLoader(
                ds, batch_size=data_cfg["batch_size"], shuffle=(split == "train"),
                num_workers=data_cfg["num_workers"],
            )

    elif split_mode == "pseudo_patient_level":
        warnings.warn(
            "split_mode='pseudo_patient_level' đang dùng PSEUDO-PATIENT ID GIẢ LẬP "
            "trên dữ liệu tạm. Đây là bước KIỂM THỬ CODE, không phải kết quả khoa học "
            "về generalization thật. Xem README.md / docs/proposal_summary.md."
        )
        # Gộp toàn bộ ảnh từ 3 thư mục có sẵn lại rồi tự chia lại theo pseudo-group,
        # để mô phỏng tình huống patient-level split thực sự (không dùng chia sẵn nữa)
        all_samples = []
        for split in ["train", "valid", "test"]:
            all_samples += _list_image_folder_samples(os.path.join(root_dir, split), class_names)

        all_samples = _assign_pseudo_patient_ids(
            all_samples,
            images_per_group=data_cfg["pseudo_patient"]["images_per_pseudo_patient"],
            seed=data_cfg["pseudo_patient"]["random_seed"],
        )

        train_val_samples, test_samples = _group_split(all_samples, test_size=0.15, seed=seed)
        train_samples, valid_samples = _group_split(train_val_samples, test_size=0.15, seed=seed)

        for split, samples in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
            transform = get_transforms(data_cfg["image_size"], split, cfg["augmentation"])
            ds = BitssStoolDataset(samples, transform)
            dataloaders[split] = DataLoader(
                ds, batch_size=data_cfg["batch_size"], shuffle=(split == "train"),
                num_workers=data_cfg["num_workers"],
            )

    elif split_mode == "patient_level":
        # Dữ liệu THẬT — patient_id thật từ metadata CSV do bệnh viện cung cấp
        metadata_csv = data_cfg.get("patient_metadata_csv")
        if not metadata_csv:
            raise ValueError(
                "split_mode='patient_level' yêu cầu data.patient_metadata_csv "
                "trỏ tới file CSV thật (xem docs/data_contract.md)."
            )
        all_samples = _load_samples_from_metadata_csv(metadata_csv)
        train_val_samples, test_samples = _group_split(all_samples, test_size=0.15, seed=seed)
        train_samples, valid_samples = _group_split(train_val_samples, test_size=0.15, seed=seed)

        for split, samples in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
            transform = get_transforms(data_cfg["image_size"], split, cfg["augmentation"])
            ds = BitssStoolDataset(samples, transform)
            dataloaders[split] = DataLoader(
                ds, batch_size=data_cfg["batch_size"], shuffle=(split == "train"),
                num_workers=data_cfg["num_workers"],
            )
    else:
        raise ValueError(f"split_mode không hợp lệ: {split_mode}")

    return dataloaders


def class_distribution_report(root_dir: str, class_names: list[str]) -> pd.DataFrame:
    """Dùng cho Giai đoạn 1 (EDA) — đếm số ảnh mỗi lớp mỗi split, phát hiện imbalance."""
    rows = []
    for split in ["train", "valid", "test"]:
        for class_name in class_names:
            class_dir = Path(root_dir) / split / class_name
            n = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
            rows.append({"split": split, "class": class_name, "count": n})
    return pd.DataFrame(rows)
