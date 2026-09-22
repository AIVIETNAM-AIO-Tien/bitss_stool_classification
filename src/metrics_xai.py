"""
Chỉ số đánh giá ĐỊNH LƯỢNG cho XAI — CHỈ dùng được khi đã có ROI annotation
thật (roi_annotation_csv trong configs/data_paths.yaml). Trên dữ liệu tạm
hiện tại chưa có ROI nên các hàm này chưa được gọi trong pipeline chính
(explain.py) — để sẵn ở đây cho Giai đoạn RQ2 khi có dữ liệu lâm sàng thật.

Tham khảo phương pháp:
    - IoU / Pointing Game: đo mức trùng khớp giữa saliency map và ROI thật.
    - Average Drop / Average Increase in Confidence: che vùng saliency quan
      trọng nhất, đo mức thay đổi confidence của model (Shap-CAM, arXiv 2208.03608).
Xem docs/related_work.md mục XAI để biết đầy đủ trích dẫn.
"""
import numpy as np
import torch


def heatmap_to_binary_mask(grayscale_cam: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Nhị phân hóa heatmap Grad-CAM (giá trị [0,1]) theo ngưỡng, để so sánh với ROI box."""
    return (grayscale_cam >= threshold).astype(np.uint8)


def roi_box_to_mask(x_min: int, y_min: int, x_max: int, y_max: int, height: int, width: int) -> np.ndarray:
    """Chuyển bounding box ROI (từ roi_annotation_csv) thành mask nhị phân cùng kích thước ảnh."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def compute_iou(saliency_mask: np.ndarray, roi_mask: np.ndarray) -> float:
    """IoU giữa vùng saliency model tập trung vào và vùng ROI thật (vùng phân).
    IoU thấp -> model có thể đang nhìn vào vùng khác (tã, nền) thay vì phân
    -> bằng chứng ủng hộ giả thuyết shortcut learning (RQ2).
    """
    intersection = np.logical_and(saliency_mask, roi_mask).sum()
    union = np.logical_or(saliency_mask, roi_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def pointing_game_hit(grayscale_cam: np.ndarray, roi_mask: np.ndarray) -> bool:
    """Pointing Game: điểm có giá trị saliency CAO NHẤT có nằm trong ROI thật không.
    Trả về True (hit) / False (miss). Tính tỷ lệ hit trên toàn bộ tập test để
    có 'Pointing Game Accuracy' — chỉ số phổ biến trong literature XAI.
    """
    max_idx = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    return bool(roi_mask[max_idx] == 1)


@torch.no_grad()
def average_drop_increase(model, input_tensor: torch.Tensor, grayscale_cam: np.ndarray,
                            target_class: int, threshold: float = 0.5, device=None) -> dict:
    """Che (mask=0) vùng saliency quan trọng nhất, đo mức thay đổi confidence
    của model đối với target_class trước/sau khi che.

        Average Drop (%)     = mean( max(0, Y_original - Y_masked) / Y_original ) * 100
        Average Increase (%) = tỷ lệ ảnh mà Y_masked > Y_original (bất thường,
                                gợi ý model không thực sự dựa vào vùng đó)

    Dùng cho một batch/nhiều ảnh rồi lấy trung bình ở cấp pipeline gọi hàm này.
    """
    model.eval()
    mask = heatmap_to_binary_mask(grayscale_cam, threshold)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if device:
        mask_tensor = mask_tensor.to(device)

    original_probs = torch.softmax(model(input_tensor), dim=1)
    y_original = original_probs[0, target_class].item()

    masked_input = input_tensor * (1 - mask_tensor)  # che vùng saliency quan trọng nhất
    masked_probs = torch.softmax(model(masked_input), dim=1)
    y_masked = masked_probs[0, target_class].item()

    drop = max(0.0, y_original - y_masked) / (y_original + 1e-8) * 100
    increased = y_masked > y_original

    return {"y_original": y_original, "y_masked": y_masked, "drop_pct": drop, "increased": increased}
