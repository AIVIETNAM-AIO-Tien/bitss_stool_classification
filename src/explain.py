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
import os
import random

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from models import build_model, get_target_layer_for_gradcam
from utils import load_config, load_checkpoint, get_device, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Sinh Grad-CAM/LIME cho model BITSS")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "lime"])
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Tổng số ảnh minh họa; mặc định lấy theo config xai.num_samples_per_class * num_classes")
    return parser.parse_args()


def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    """Chuyển tensor đã normalize về ảnh RGB [0,1] để overlay heatmap / hiển thị."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_img.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def run_gradcam(model, target_layer, dataloader, class_names, device, num_samples, output_dir, logger):
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

    # Lấy ngẫu nhiên num_samples ảnh từ tập test để minh họa
    dataset = dataloader.dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

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

    logger.info(f"Đã lưu {len(indices)} ảnh Grad-CAM vào {output_dir}/")
    logger.warning(
        "Đây là minh họa ĐỊNH TÍNH (proof-of-concept) trên dữ liệu tạm — chưa có "
        "ROI annotation thật để đánh giá ĐỊNH LƯỢNG (IoU/Pointing Game). "
        "Xem src/metrics_xai.py — sẽ dùng được ngay khi có roi_annotation_csv thật."
    )


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
    output_dir = cfg["xai"]["output_dir"]

    if args.method == "gradcam":
        target_layer = get_target_layer_for_gradcam(model, cfg["model"]["backbone"])
        run_gradcam(model, target_layer, dataloaders["test"], class_names, device,
                    num_samples, output_dir, logger)
    elif args.method == "lime":
        run_lime(model, dataloaders["test"], class_names, device, num_samples, output_dir, logger)


if __name__ == "__main__":
    main()
