"""
Trích xuất và trực quan hóa các mẫu bị dự đoán sai — Giai đoạn Error Analysis
(Chương 5.3.2 trong báo cáo).

Vì BITSS là thang đo ORDINAL, lỗi được phân loại theo độ lệch |y_true - y_pred|:
    - lệch 1 mức (liền kề, ví dụ Type_3 -> Type_4): lỗi nhẹ, dễ chấp nhận lâm sàng
    - lệch >= 3 mức (ví dụ Type_1 -> Type_5): lỗi nghiêm trọng, ưu tiên phân tích

Ví dụ chạy:
    !python src/error_analysis.py --config configs/default.yaml \
        --checkpoint outputs/checkpoints/best.pt --min_gap 2 --max_display 12
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from models import build_model
from utils import load_config, load_checkpoint, get_device, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Phân tích lỗi dự đoán trên tập test")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--min_gap", type=int, default=1,
                         help="Chỉ liệt kê lỗi có |y_true - y_pred| >= min_gap. "
                              "Đặt 3 để chỉ xem lỗi NGHIÊM TRỌNG (lệch xa) theo tinh thần ordinal.")
    parser.add_argument("--max_display", type=int, default=12,
                         help="Số ảnh tối đa hiển thị trong lưới ảnh minh họa")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Mặc định: <evaluate.report_dir>/error_analysis")
    return parser.parse_args()


def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_img.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Thu thập toàn bộ dự đoán kèm confidence — dùng cho cả bảng lỗi và error grid."""
    model.eval()
    rows = []
    for images, labels, paths in dataloader:
        images_dev = images.to(device)
        logits = model(images_dev)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

        for i in range(len(labels)):
            rows.append({
                "image_path": paths[i],
                "y_true": int(labels[i]),
                "y_pred": int(preds[i].cpu()),
                "confidence": float(confidences[i].cpu()),
                "gap": abs(int(labels[i]) - int(preds[i].cpu())),
            })
    return pd.DataFrame(rows)


def plot_error_grid(error_df: pd.DataFrame, dataloader, class_names: list[str],
                     max_display: int, save_path: str):
    """Vẽ lưới ảnh lỗi, sắp xếp theo gap giảm dần (lỗi nghiêm trọng nhất trước)."""
    if len(error_df) == 0:
        print("Không có lỗi nào khớp điều kiện lọc — không vẽ lưới ảnh.")
        return

    error_df = error_df.sort_values("gap", ascending=False).head(max_display)
    path_to_tensor = {dataloader.dataset.samples[i].image_path: i for i in range(len(dataloader.dataset))}

    n = len(error_df)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, error_df.iterrows()):
        idx = path_to_tensor[row["image_path"]]
        image_tensor, _, _ = dataloader.dataset[idx]
        img = denormalize(image_tensor)
        ax.imshow(img)
        ax.set_title(
            f"Thật: {class_names[row['y_true']]} | Đoán: {class_names[row['y_pred']]}\n"
            f"(lệch {row['gap']} mức, conf={row['confidence']:.2f})",
            fontsize=9, color="red" if row["gap"] >= 3 else "black",
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("error_analysis", cfg["train"]["log_dir"])
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["evaluate"]["checkpoint"]
    ckpt = load_checkpoint(checkpoint_path, map_location=device)

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg["model"], cfg["data"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    class_names = cfg["data"]["class_names"]
    output_dir = args.output_dir or os.path.join(cfg["evaluate"]["report_dir"], "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Đang thu thập dự đoán trên tập test...")
    df = collect_predictions(model, dataloaders["test"], device)

    df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)

    error_df = df[df["y_true"] != df["y_pred"]].copy()
    filtered_df = error_df[error_df["gap"] >= args.min_gap].copy()
    filtered_df.to_csv(os.path.join(output_dir, "errors_filtered.csv"), index=False)

    n_total = len(df)
    n_errors = len(error_df)
    n_severe = len(error_df[error_df["gap"] >= 3])

    logger.info(f"Tổng số ảnh test: {n_total}")
    logger.info(f"Tổng số lỗi (y_true != y_pred): {n_errors} ({100*n_errors/n_total:.1f}%)")
    logger.info(f"Lỗi NGHIÊM TRỌNG (lệch >= 3 mức): {n_severe} ({100*n_severe/n_total:.1f}% tổng số, "
                f"{100*n_severe/max(n_errors,1):.1f}% trong số lỗi)")
    logger.info(f"Lỗi khớp điều kiện lọc (gap >= {args.min_gap}): {len(filtered_df)}")

    # Phân bố gap của lỗi — hữu ích để đưa vào bảng 5.3.1 mở rộng
    gap_distribution = error_df["gap"].value_counts().sort_index()
    logger.info(f"Phân bố độ lệch (gap) của các lỗi:\n{gap_distribution}")
    gap_distribution.to_csv(os.path.join(output_dir, "gap_distribution.csv"))

    plot_error_grid(
        filtered_df, dataloaders["test"], class_names,
        max_display=args.max_display,
        save_path=os.path.join(output_dir, "error_grid.png"),
    )

    logger.info(f"Đã lưu kết quả error analysis vào {output_dir}/")
    logger.info(
        "Dùng error_grid.png + errors_filtered.csv cho mục 5.3.2 báo cáo. "
        "Kết hợp với Grad-CAM (src/explain.py) trên cùng các ảnh này cho mục 5.3.3 "
        "(liên hệ Error Analysis với XAI)."
    )


if __name__ == "__main__":
    main()
