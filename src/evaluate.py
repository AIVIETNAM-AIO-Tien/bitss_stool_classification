"""
Đánh giá model đã train trên tập test.
Xuất: bảng metrics + confusion matrix (lưu ảnh vào outputs/figures).

Ví dụ chạy:
    !python src/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
"""
import argparse
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
)

from dataset import build_dataloaders
from models import build_model
from utils import load_config, load_checkpoint, get_device, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BITSS stool classification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Mặc định lấy từ evaluate.checkpoint trong config")
    return parser.parse_args()


@torch.no_grad()
def predict_all(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    for images, labels, paths in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
    return np.array(all_labels), np.array(all_preds), all_paths


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — BITSS Type 1-7")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("evaluate", cfg["train"]["log_dir"])
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["evaluate"]["checkpoint"]
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    logger.info(f"Đã nạp checkpoint từ epoch {ckpt['epoch']} ({checkpoint_path})")

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg["model"], cfg["data"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, paths = predict_all(model, dataloaders["test"], device)
    class_names = cfg["data"]["class_names"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa_quadratic": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "split_mode": cfg["data"]["split_mode"],
        "data_profile": cfg.get("_active_data_profile", "unknown"),
    }

    logger.info(f"Test metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    report_dir = cfg["evaluate"]["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(report_dir, "confusion_matrix.png"))
    logger.info(f"Đã lưu kết quả vào {report_dir}/")

    if cfg["data"]["split_mode"] == "pseudo_patient_level":
        logger.warning(
            "NHẮC LẠI: kết quả này chạy trên pseudo-patient split (dữ liệu tạm) — "
            "chỉ để kiểm thử code, không dùng để so sánh khoa học với Setup A."
        )


if __name__ == "__main__":
    main()
