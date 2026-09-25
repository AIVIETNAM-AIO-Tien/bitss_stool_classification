"""
Tiện ích dùng chung: set seed, đọc config yaml, lưu/đọc checkpoint, logging.
Tách riêng để train.py / evaluate.py / explain.py không lặp code.
"""
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho reproducibility (torch, numpy, python random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Đọc file config yaml chính, đồng thời merge với data_paths.yaml cùng thư mục
    nếu tồn tại, để tách biệt đường dẫn dữ liệu khỏi tham số huấn luyện.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_paths_path = Path(config_path).parent / "data_paths.yaml"
    if data_paths_path.exists():
        with open(data_paths_path, "r", encoding="utf-8") as f:
            paths_cfg = yaml.safe_load(f)
        active = paths_cfg["active_profile"]
        profile = paths_cfg["profiles"][active]
        # Ghi đè các trường liên quan tới đường dẫn trong cfg["data"]
        cfg["data"]["root_dir"] = profile["root_dir"]
        cfg["data"]["patient_metadata_csv"] = profile.get("metadata_csv")
        cfg["xai"]["roi_annotation_csv"] = profile.get("roi_annotation_csv")
        cfg["_active_data_profile"] = active

    return cfg


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Logger đơn giản, ghi ra cả console và file — hữu ích khi chạy trên Colab
    vì output cell có thể bị mất khi session bị ngắt.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:  # tránh add handler trùng khi gọi lại nhiều lần (Jupyter)
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> str:
    """Lưu checkpoint. Gọi thường xuyên (save_every_n_epochs) để phòng Colab
    ngắt session giữa chừng — mất tiến độ train là rủi ro đã nêu trong kế hoạch.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, map_location=None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại {path}. "
            "Đã chạy src/train.py trước đó chưa?"
        )
    return torch.load(path, map_location=map_location)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
