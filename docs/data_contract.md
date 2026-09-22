# Data Contract — Schema dữ liệu THẬT cần bệnh viện cung cấp

Tài liệu này là "hợp đồng" giữa pipeline code và dữ liệu lâm sàng thật, để
khi dữ liệu về, việc tích hợp vào pipeline chỉ là thao tác cấu hình, không
phải viết lại code.

## 1. Cấu trúc thư mục ảnh

```
data/clinical_v1/
├── images/
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   └── ...
├── metadata.csv
└── roi.csv                # optional ở giai đoạn đầu, bắt buộc cho đánh giá XAI định lượng (RQ2)
```

Không bắt buộc giữ cấu trúc `Type_1..7` theo thư mục như dữ liệu tạm — với
dữ liệu thật, nhãn và mọi thông tin khác nằm trong `metadata.csv`, ảnh có
thể để phẳng trong 1 thư mục `images/`.

## 2. `metadata.csv` — bắt buộc

| Cột | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `image_path` | string | ✅ | Đường dẫn tương đối tới ảnh (từ `data/clinical_v1/`) |
| `patient_id` | string | ✅ | Mã ẩn danh **duy nhất cho mỗi trẻ** — KHÔNG dùng tên/số hồ sơ bệnh án thật, dùng mã hash/số thứ tự đã ẩn danh hóa. **Đây là trường quan trọng nhất** — thiếu trường này thì không chạy được Setup B (patient-level split), toàn bộ RQ1 mất giá trị khoa học. |
| `label` | int (0–6) | ✅ | Nhãn BITSS đã gán, mã hóa 0=Type_1 ... 6=Type_7 (khớp thứ tự `class_names` trong `configs/default.yaml`) |
| `site` | string | Khuyến nghị | Cơ sở y tế thu thập (nếu có nhiều điểm) — dùng cho phân tích cross-site sau này |
| `device` | string | Khuyến nghị | Loại thiết bị/điện thoại chụp — một trong các yếu tố nghi ngờ gây shortcut (theo Hill et al. 2024) |
| `diaper_type` | string | Khuyến nghị | Loại tã — yếu tố nghi ngờ gây shortcut khác cần kiểm soát |
| `collection_date` | date | Khuyến nghị | Ngày chụp — hỗ trợ chia theo thời gian nếu cần kiểm tra temporal drift |
| `annotator_id` | string | Khuyến nghị | Người gán nhãn (nếu có nhiều người, phục vụ tính inter-rater agreement) |

## 3. `roi.csv` — cần cho đánh giá XAI định lượng (RQ2), có thể bổ sung sau

| Cột | Kiểu | Mô tả |
|---|---|---|
| `image_path` | string | Khớp với `metadata.csv` |
| `x_min`, `y_min`, `x_max`, `y_max` | int | Bounding box (pixel) khoanh vùng phân trong ảnh |

**Khuyến nghị số lượng annotate**: tối thiểu ~50–100 ảnh trải đều 7 lớp
(xem `docs/proposal_summary.md` mục điều kiện tiên quyết RQ2) — đủ để tính
IoU/Pointing Game có ý nghĩa mà không tốn quá nhiều thời gian thủ công với
1 người thực hiện.

## 4. Yêu cầu đạo đức đi kèm dữ liệu (không phải file, nhưng bắt buộc trước khi nhận dữ liệu)

- Xác nhận đã có **chấp thuận đạo đức (IRB)** từ bệnh viện.
- Xác nhận đã có **đồng thuận (consent)** từ phụ huynh/người giám hộ cho từng ảnh.
- Ảnh đã được **ẩn danh hóa** (loại bỏ EXIF chứa GPS/thiết bị cá nhân hoá, không có khuôn mặt/thông tin nhận diện trẻ) trước khi rời khỏi cơ sở y tế.

## 5. Việc cần làm trong code khi dữ liệu thật về

1. Đặt ảnh + `metadata.csv` (+ `roi.csv` nếu có) vào `data/clinical_v1/` theo đúng schema trên.
2. Sửa `configs/data_paths.yaml`: đổi `active_profile: "proxy"` → `active_profile: "clinical_v1"`.
3. Sửa `configs/default.yaml`: đổi `data.split_mode: "patient_level"`.
4. Chạy lại `src/train.py` — không cần sửa bất kỳ dòng code nào khác.
5. Nếu đã có `roi.csv`: dùng `src/metrics_xai.py` để tính IoU/Pointing Game định lượng cho Grad-CAM (hiện các hàm này đã viết sẵn nhưng chưa được gọi trong `explain.py` vì chưa có ROI thật).
