# Tóm tắt khung đề án

## Đề tài
Xây dựng mô hình học sâu và AI giải thích (XAI) nhận diện, phân loại hình
thái phân trẻ em theo thang đo Brussels (BITSS) hỗ trợ chẩn đoán lâm sàng.

## Bối cảnh & khoảng trống nghiên cứu
Xuất phát từ hai công trình:
- **Ludwig et al. (2021)**, JPGN — 2,687 ảnh/96 trẻ, chia train/test ở
  **cấp độ ảnh** → nguy cơ data leakage, model có thể học shortcut (tã,
  nền, ánh sáng) thay vì đặc điểm stool consistency thật.
- **Xiao et al. (2023)**, Acta Paediatrica — chỉ báo cáo hiệu năng model,
  **không có phân tích interpretability**.

## Câu hỏi nghiên cứu (RQ)

| RQ | Nội dung |
|---|---|
| RQ1 | Model có thực sự học đặc trưng stool consistency, hay học shortcut? Khả năng tổng quát hóa trên bệnh nhân mới? |
| RQ1a | So sánh hiệu năng Setup A (image-level split) vs Setup B (patient-level split) |
| RQ2 | Các phương pháp XAI (Grad-CAM, LIME) có phát hiện được shortcut learning không? |
| RQ2a | Vùng XAI làm nổi bật có trùng với ROI thật (vùng phân) hay không (IoU/Pointing Game) |

## Điều kiện tiên quyết chính (tóm tắt)

- **RQ1/RQ1a**: bắt buộc có `patient_id` thật trong metadata, tối thiểu
  ~30–50 trẻ, mỗi trẻ ≥ 3–5 ảnh.
- **RQ2/RQ2a**: cần ROI annotation thủ công tối thiểu ~50–100 ảnh.
- Compute: Google Colab (free/pro) — đủ cho MobileNetV2/ResNet18/EfficientNet-B0,
  KHÔNG đủ để chạy SHAP trên ảnh ở quy mô lớn (ưu tiên Grad-CAM/LIME).
- Nhân lực: 1 người thực hiện → điểm nghẽn chính là **thời gian gán nhãn +
  ROI annotation thủ công**, không phải compute.

## Trạng thái hiện tại (giai đoạn dữ liệu tạm)
Repo này đang chạy trên dataset proxy (`Type_1..7` theo ImageFolder, không
có `patient_id` thật) để kiểm thử pipeline kỹ thuật trong lúc chờ IRB từ
bệnh viện đã hợp tác. Xem `README.md` và `data_contract.md` để biết chi
tiết ranh giới giữa "kết quả kiểm thử code" và "kết quả khoa học chính thức".

Xem `related_work.md` cho bảng đầy đủ paper liên quan theo từng phần, và
`ta_questions.md` cho danh sách câu hỏi cần làm rõ với TA.
