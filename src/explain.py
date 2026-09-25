"""
Sinh giải thích XAI cho model đã train — Giai đoạn 5 trong kế hoạch.

Ưu tiên Grad-CAM (nhẹ, ổn định trên ảnh theo review PMC12383817 — xem
docs/related_work.md), LIME là tùy chọn đối chiếu chéo. SHAP KHÔNG bật mặc
định vì tốn compute trên Colab và fidelity thấp hơn trên ảnh — chỉ bật thủ
công nếu cần proof-of-concept trên vài ảnh.

Ví dụ chạy:
    !python src/explain.py --config configs/default.yaml \
        --checkpoint outputs/checkpoints/best.pt --method gradcam --num_samples 8
"""
import argparse
import json
import os
import random

import pandas as pd
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from models import build_model, get_target_layer_for_gradcam
from metrics_xai import (
    heatmap_to_binary_mask, roi_box_to_mask, compute_iou,
    pointing_game_hit, average_drop_increase,
)
from utils import load_config, load_checkpoint, get_device, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Sinh Grad-CAM/LIME cho model BITSS")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "lime"])
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Tổng số ảnh minh họa; mặc định lấy theo config xai.num_samples_per_class * num_classes")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Ghi đè xai.output_dir — hữu ích khi so sánh Grad-CAM giữa "
                              "nhiều setup (Setup A vs B), tránh ghi đè kết quả lẫn nhau")
    return parser.parse_args()


def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    """Chuyển tensor đã normalize về ảnh RGB [0,1] để overlay heatmap / hiển thị."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_img.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def run_gradcam(model, target_layer, dataloader, class_names, device, num_samples, output_dir, logger):
    """Sinh Grad-CAM cho num_samples ảnh ngẫu nhiên, lưu ảnh minh họa, và trả về
    danh sách record {image_path, label, grayscale_cam} để hàm gọi tiếp (đánh giá
    định lượng nếu có ROI, hoặc error_analysis) tái sử dụng mà không phải chạy CAM 2 lần.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        raise ImportError(
            "Thiếu thư viện pytorch-grad-cam. Cài bằng: pip install grad-cam "
            "(xem requirements.txt)"
        )

    cam = GradCAM(model=model, target_layers=[target_layer])
    os.makedirs(output_dir, exist_ok=True)

    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    records = []
    for i, idx in enumerate(indices):
        image_tensor, label, image_path = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = denormalize(image_tensor)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Gốc — {class_names[label]}")
        axes[0].axis("off")
        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"gradcam_{i:03d}_{class_names[label]}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        records.append({
            "image_path": image_path,
            "label": label,
            "grayscale_cam": grayscale_cam,
            "input_tensor": input_tensor,
        })

    logger.info(f"Đã lưu {len(indices)} ảnh Grad-CAM vào {output_dir}/")
    logger.warning(
        "Minh họa ĐỊNH TÍNH (proof-of-concept) trên dữ liệu tạm — nếu chưa cấu hình "
        "xai.roi_annotation_csv, phần đánh giá ĐỊNH LƯỢNG (IoU/Pointing Game) sẽ bị bỏ qua. "
        "Xem docs/data_contract.md để biết cách cấu hình khi có ROI thật."
    )
    return records


def run_quantitative_xai_eval(model, records, roi_csv_path, device, output_dir, logger):
    """Đánh giá ĐỊNH LƯỢNG Grad-CAM bằng metrics_xai.py — CHỈ chạy được khi có
    roi_annotation_csv (dữ liệu thật). Trên proxy data hàm này được gọi nhưng sẽ
    tự bỏ qua và log lý do, đúng tinh thần 'để trống + nêu rõ điều kiện' của báo cáo.
    """
    if not roi_csv_path or not os.path.exists(roi_csv_path):
        logger.warning(
            "Bỏ qua đánh giá định lượng XAI (IoU/Pointing Game/Average Drop): "
            "chưa có xai.roi_annotation_csv hợp lệ trong config. "
            "Đây là điều kiện dữ liệu thật theo docs/data_contract.md, "
            "hiện KHÔNG đáp ứng được trên dataset proxy — mục 5.3.4/2.6 báo cáo."
        )
        return None

    roi_df = pd.read_csv(roi_csv_path)
    roi_lookup = {row["image_path"]: row for _, row in roi_df.iterrows()}

    rows = []
    for rec in records:
        roi_row = roi_lookup.get(rec["image_path"])
        if roi_row is None:
            continue  # ảnh này chưa có ROI annotation, bỏ qua khi tính trung bình

        h, w = rec["grayscale_cam"].shape
        roi_mask = roi_box_to_mask(
            int(roi_row["x_min"]), int(roi_row["y_min"]),
            int(roi_row["x_max"]), int(roi_row["y_max"]), h, w,
        )
        saliency_mask = heatmap_to_binary_mask(rec["grayscale_cam"], threshold=0.5)

        iou = compute_iou(saliency_mask, roi_mask)
        hit = pointing_game_hit(rec["grayscale_cam"], roi_mask)
        drop_info = average_drop_increase(
            model, rec["input_tensor"], rec["grayscale_cam"],
            target_class=rec["label"], threshold=0.5, device=device,
        )

        rows.append({
            "image_path": rec["image_path"], "label": rec["label"],
            "iou": iou, "pointing_game_hit": hit,
            "confidence_drop_pct": drop_info["drop_pct"],
            "confidence_increased_after_masking": drop_info["increased"],
        })

    if not rows:
        logger.warning("roi_annotation_csv tồn tại nhưng không khớp ảnh nào trong mẫu Grad-CAM hiện tại.")
        return None

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "xai_quantitative_metrics.csv"), index=False)

    summary = {
        "n_images_evaluated": len(df),
        "mean_iou": float(df["iou"].mean()),
        "pointing_game_accuracy": float(df["pointing_game_hit"].mean()),
        "mean_confidence_drop_pct": float(df["confidence_drop_pct"].mean()),
        "pct_confidence_increased_after_masking": float(df["confidence_increased_after_masking"].mean()) * 100,
    }
    with open(os.path.join(output_dir, "xai_quantitative_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Đánh giá định lượng XAI (RQ2a) — {json.dumps(summary, indent=2, ensure_ascii=False)}")
    logger.info(
        "IoU/Pointing Game THẤP hoặc % confidence tăng sau khi che vùng saliency CAO "
        "là bằng chứng ủng hộ giả thuyết shortcut learning (mô hình không thực sự "
        "dựa vào vùng phân để ra quyết định) — dùng trực tiếp cho Chương 5.3.3/5.5 báo cáo."
    )
    return summary


def run_lime(model, dataloader, class_names, device, num_samples, output_dir, logger):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        raise ImportError("Thiếu thư viện lime/scikit-image. Cài: pip install lime scikit-image")

    model.eval()

    def predict_fn(images_np):
        """LIME cần hàm nhận batch ảnh numpy [N,H,W,3] trong [0,255] và trả về xác suất."""
        batch = []
        for img in images_np:
            t = torch.tensor(img / 255.0).permute(2, 0, 1).float()
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            t = (t - mean) / std
            batch.append(t)
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    os.makedirs(output_dir, exist_ok=True)

    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    for i, idx in enumerate(indices):
        image_tensor, label, image_path = dataset[idx]
        rgb_img = (denormalize(image_tensor) * 255).astype(np.uint8)

        explanation = explainer.explain_instance(
            rgb_img, predict_fn, top_labels=1, hide_color=0, num_samples=200  # 200: cân đối tốc độ trên Colab
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f"LIME — {class_names[label]}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_{i:03d}_{class_names[label]}.png"), dpi=150)
        plt.close()

    logger.info(f"Đã lưu {len(indices)} ảnh LIME vào {output_dir}/")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("explain", cfg["train"]["log_dir"])
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["evaluate"]["checkpoint"]
    ckpt = load_checkpoint(checkpoint_path, map_location=device)

    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg["model"], cfg["data"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_names = cfg["data"]["class_names"]
    num_samples = args.num_samples or (cfg["xai"]["num_samples_per_class"] * len(class_names))
    output_dir = args.output_dir or cfg["xai"]["output_dir"]

    if args.method == "gradcam":
        target_layer = get_target_layer_for_gradcam(model, cfg["model"]["backbone"])
        records = run_gradcam(model, target_layer, dataloaders["test"], class_names, device,
                               num_samples, output_dir, logger)
        run_quantitative_xai_eval(
            model, records, cfg["xai"].get("roi_annotation_csv"),
            device, output_dir, logger,
        )
    elif args.method == "lime":
        run_lime(model, dataloaders["test"], class_names, device, num_samples, output_dir, logger)


if __name__ == "__main__":
    main()
