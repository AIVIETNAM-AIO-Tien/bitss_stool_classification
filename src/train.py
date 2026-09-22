"""
CLI training script.

Ví dụ chạy trên Colab:
    !python src/train.py --config configs/default.yaml --split_mode image_level
    !python src/train.py --config configs/default.yaml --split_mode pseudo_patient_level

Lưu checkpoint thường xuyên (save_every_n_epochs) để giảm rủi ro mất tiến độ
khi Colab free tier ngắt session giữa chừng (đã nêu trong kế hoạch Giai đoạn 3).
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from tqdm import tqdm

from dataset import build_dataloaders
from models import build_model, count_trainable_params
from utils import set_seed, load_config, get_logger, save_checkpoint, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train BITSS stool classification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split_mode", type=str, default=None,
                         choices=["image_level", "pseudo_patient_level", "patient_level"],
                         help="Ghi đè data.split_mode trong config nếu truyền vào")
    return parser.parse_args()


def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(dataloader, desc="train" if train else "eval", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
    }
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.split_mode:
        cfg["data"]["split_mode"] = args.split_mode

    set_seed(cfg.get("seed", 42))
    logger = get_logger("train", cfg["train"]["log_dir"])
    device = get_device()
    logger.info(f"Device: {device} | split_mode: {cfg['data']['split_mode']} | "
                f"data profile: {cfg.get('_active_data_profile', 'unknown')}")

    if cfg["data"]["split_mode"] == "pseudo_patient_level":
        logger.warning(
            "Đang chạy với PSEUDO-PATIENT ID GIẢ LẬP trên dữ liệu tạm. "
            "Kết quả CHỈ dùng để kiểm thử logic code, KHÔNG phải kết luận khoa học."
        )

    dataloaders = build_dataloaders(cfg)
    logger.info(f"Số lượng ảnh: train={len(dataloaders['train'].dataset)}, "
                f"valid={len(dataloaders['valid'].dataset)}, test={len(dataloaders['test'].dataset)}")

    model = build_model(cfg["model"], cfg["data"]["num_classes"]).to(device)
    trainable, total = count_trainable_params(model)
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.1f}%) — backbone={cfg['model']['backbone']}, "
                f"freeze_backbone={cfg['model']['freeze_backbone']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = None
    if cfg["train"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    best_metric = -float("inf")
    epochs_no_improve = 0
    early_stop_key = cfg["train"]["early_stopping_metric"].replace("val_", "")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, dataloaders["train"], criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, dataloaders["valid"], criterion, optimizer, device, train=False)
        if scheduler:
            scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{cfg['train']['epochs']} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} train_kappa={train_metrics['kappa']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} val_kappa={val_metrics['kappa']:.4f}"
        )

        current_metric = val_metrics[early_stop_key]
        is_best = current_metric > best_metric

        if epoch % cfg["train"]["save_every_n_epochs"] == 0 or is_best:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            }
            save_checkpoint(state, cfg["train"]["checkpoint_dir"], f"epoch_{epoch}.pt")
            if is_best:
                save_checkpoint(state, cfg["train"]["checkpoint_dir"], "best.pt")
                logger.info(f"  -> Checkpoint mới tốt nhất (val_{early_stop_key}={current_metric:.4f})")

        if is_best:
            best_metric = current_metric
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg["train"]["early_stopping_patience"]:
            logger.info(f"Early stopping tại epoch {epoch} (không cải thiện "
                        f"{cfg['train']['early_stopping_patience']} epoch liên tiếp)")
            break

    logger.info(f"Hoàn tất training. Best val_{early_stop_key} = {best_metric:.4f}. "
                f"Checkpoint tại {os.path.join(cfg['train']['checkpoint_dir'], 'best.pt')}")


if __name__ == "__main__":
    main()
