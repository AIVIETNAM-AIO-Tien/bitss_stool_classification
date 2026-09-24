"""
Vẽ đường cong huấn luyện (loss, Kappa, F1-macro theo epoch) — Chương 5.2.4.

Đọc từ file CSV history do src/train.py sinh ra tự động tại
outputs/logs/history_<split_mode>.csv (đáng tin cậy hơn parse text log).

Ví dụ chạy:
    !python src/plot_training_curves.py --config configs/default.yaml --split_mode image_level

Có thể truyền nhiều --split_mode để so sánh đường cong giữa các thí nghiệm
trên cùng 1 hình (hữu ích khi so sánh Setup A vs Setup B pseudo — mục 5.1.2).
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ đường cong huấn luyện từ history CSV")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--runs", type=str, nargs="+",
                         default=None,
                         help="Danh sách '<backbone>_<split_mode>' để so sánh trên cùng "
                              "biểu đồ, ví dụ: --runs mobilenet_v2_image_level "
                              "mobilenet_v2_pseudo_patient_level. Mặc định: tự suy ra "
                              "từ model.backbone + data.split_mode trong config.")
    parser.add_argument("--output", type=str, default=None,
                         help="Mặc định: <evaluate.report_dir>/training_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    log_dir = cfg["train"]["log_dir"]
    output_path = args.output or os.path.join(cfg["evaluate"]["report_dir"], "training_curves.png")

    runs = args.runs or [f"{cfg['model']['backbone']}_{cfg['data']['split_mode']}"]

    histories = {}
    for run_name in runs:
        path = os.path.join(log_dir, f"history_{run_name}.csv")
        if not os.path.exists(path):
            print(f"⚠️  Không tìm thấy {path} — bỏ qua. "
                  f"Đã chạy `train.py` với backbone/split_mode tương ứng '{run_name}' chưa?")
            continue
        histories[run_name] = pd.read_csv(path)

    if not histories:
        raise FileNotFoundError(
            "Không có file history nào được tìm thấy. Chạy src/train.py trước."
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics_to_plot = [
        ("loss", "Loss", axes[0]),
        ("kappa", "Cohen's Kappa (quadratic)", axes[1]),
        ("f1_macro", "F1-macro", axes[2]),
    ]

    for metric_key, title, ax in metrics_to_plot:
        for run_name, df in histories.items():
            ax.plot(df["epoch"], df[f"train_{metric_key}"], "--", label=f"{run_name} (train)", alpha=0.6)
            ax.plot(df["epoch"], df[f"val_{metric_key}"], "-", label=f"{run_name} (val)")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ đường cong huấn luyện vào {output_path}")

    # In nhanh epoch tốt nhất mỗi history — hữu ích đối chiếu với early stopping log
    for run_name, df in histories.items():
        best_row = df.loc[df["val_kappa"].idxmax()]
        print(f"[{run_name}] Epoch tốt nhất theo val_kappa: "
              f"epoch={int(best_row['epoch'])}, val_kappa={best_row['val_kappa']:.4f}, "
              f"val_loss={best_row['val_loss']:.4f}")


if __name__ == "__main__":
    main()
