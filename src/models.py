"""
Factory tạo model backbone cho bài toán 7-class BITSS classification.

Backbone mặc định: MobileNetV2 — nhẹ, chạy tốt trên Colab free tier (T4 GPU),
đồng thời là backbone bài báo gốc Ludwig et al. (2021) sử dụng, thuận tiện
đối chiếu phương pháp luận khi có dữ liệu thật (xem docs/related_work.md).
"""
import torch
import torch.nn as nn
from torchvision import models


def _freeze_all_but_last_n_blocks(model: nn.Module, backbone_name: str, n: int) -> None:
    """Đóng băng phần lớn backbone, chỉ để hở n block cuối + classifier head
    cho fine-tune — cân bằng giữa transfer learning hiệu quả và tránh overfit
    trên dataset nhỏ (đúng tinh thần điều kiện compute hạn chế trên Colab).
    """
    for param in model.parameters():
        param.requires_grad = False

    if backbone_name == "mobilenet_v2":
        blocks = list(model.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
    elif backbone_name == "resnet18":
        layer_groups = [model.layer1, model.layer2, model.layer3, model.layer4]
        for layer in layer_groups[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif backbone_name == "efficientnet_b0":
        blocks = list(model.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    # Classifier head luôn được huấn luyện
    for name, module in model.named_modules():
        if name in ("classifier", "fc"):
            for param in module.parameters():
                param.requires_grad = True


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    backbone_name = model_cfg["backbone"]
    pretrained = model_cfg.get("pretrained", True)
    dropout = model_cfg.get("dropout", 0.3)

    if backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Backbone '{backbone_name}' chưa được hỗ trợ. "
            f"Lựa chọn hợp lệ: mobilenet_v2, resnet18, efficientnet_b0"
        )

    if model_cfg.get("freeze_backbone", True):
        _freeze_all_but_last_n_blocks(
            model, backbone_name, n=model_cfg.get("unfreeze_last_n_blocks", 2)
        )

    return model


def get_target_layer_for_gradcam(model: nn.Module, backbone_name: str):
    """Trả về layer phù hợp để Grad-CAM hook vào, tùy backbone.
    Dùng khi config xai.target_layer == 'auto'.
    """
    if backbone_name == "mobilenet_v2":
        return model.features[-1]
    elif backbone_name == "resnet18":
        return model.layer4[-1]
    elif backbone_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Chưa định nghĩa target layer Grad-CAM cho backbone '{backbone_name}'")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Trả về (số param có thể train, tổng số param) — hữu ích để log/kiểm tra
    freeze_backbone có hoạt động đúng như kỳ vọng không."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
