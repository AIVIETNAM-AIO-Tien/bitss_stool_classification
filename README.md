# BITSS Stool Classification — Deep Learning + XAI

Đồ án môn học: Xây dựng mô hình học sâu và AI giải thích (XAI) nhận diện, phân
loại hình thái phân trẻ em theo thang đo Brussels (BITSS) hỗ trợ chẩn đoán
lâm sàng.

## ⚠️ Trạng thái hiện tại: PIPELINE PROTOTYPE trên dữ liệu TẠM (proxy dataset)

Repo này đang ở giai đoạn **kiểm thử pipeline kỹ thuật** trong lúc chờ dữ
liệu lâm sàng thật (đang chờ IRB/hợp tác bệnh viện). Điều đó có nghĩa:

- Dataset hiện tại (`data/train|valid|test/Type_{1..7}/*.jpg`) **không có
  ID bệnh nhân thật** → mọi kết quả liên quan đến "patient-level split"
  hiện chỉ chạy trên **pseudo-patient ID giả lập** để kiểm thử logic code,
  **không dùng để kết luận khoa học về generalization**.
- Grad-CAM/XAI ở bước này là **proof-of-concept**, chưa có ROI ground-truth
  để đánh giá định lượng (IoU, Pointing Game) — việc đó chỉ thực hiện khi
  có dữ liệu thật + annotation thật.
- Khi dữ liệu thật sẵn sàng: chỉ cần (1) thay đường dẫn dữ liệu, (2) cung
  cấp cột `patient_id` thật trong metadata, (3) thêm ROI annotation. Phần
  còn lại của pipeline (model, training loop, XAI, evaluation) tái sử dụng
  gần như nguyên vẹn — xem `docs/data_contract.md`.

Xem chi tiết khung đề án, câu hỏi RQ, related work trong `docs/`.

## Cấu trúc repo

```
bitss-stool-classification/
├── configs/                # File cấu hình (yaml) — không sửa code khi đổi tham số
│   ├── default.yaml
│   └── data_paths.yaml
├── data/                   # Dataset (KHÔNG commit ảnh thật lên git — xem .gitignore)
│   ├── train/Type_1 ... Type_7/*.jpg
│   ├── valid/Type_1 ... Type_7/*.jpg
│   └── test/Type_1 ... Type_7/*.jpg
├── docs/
│   ├── proposal_summary.md     # Tóm tắt khung đề án, RQ1/RQ2
│   ├── related_work.md         # Bảng paper liên quan theo từng phần
│   ├── ta_questions.md         # Câu hỏi cho buổi trao đổi TA
│   └── data_contract.md        # "Hợp đồng" schema dữ liệu thật cần có
├── notebooks/
│   ├── 01_eda.ipynb             # Giai đoạn 1: khám phá dữ liệu
│   ├── 02_train_baseline.ipynb  # Giai đoạn 3: train Setup A
│   ├── 03_split_experiment.ipynb# Giai đoạn 4: Setup A vs pseudo-patient B
│   └── 04_xai_demo.ipynb        # Giai đoạn 5: Grad-CAM/LIME demo
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Dataset/DataLoader, split_mode image|pseudo_patient|patient
│   ├── models.py            # Backbone factory (MobileNetV2/ResNet18/EfficientNet)
│   ├── train.py             # Training loop CLI
│   ├── evaluate.py          # Eval: accuracy, F1-macro, Cohen's Kappa, confusion matrix
│   ├── explain.py           # Grad-CAM / LIME wrapper
│   ├── metrics_xai.py       # IoU/Pointing Game/Average Drop (dùng khi có ROI thật)
│   └── utils.py             # seed, checkpoint I/O, logging helpers
├── tests/
│   └── test_dataset.py      # Unit test nhanh cho data pipeline (chạy trên vài ảnh mẫu)
├── outputs/                 # checkpoints, logs, figures (gitignore phần lớn)
├── requirements.txt
├── .gitignore
└── README.md
```

## Cách chạy nhanh (Google Colab)

```python
!git clone <repo-url>
%cd bitss-stool-classification
!pip install -r requirements.txt -q

# Giai đoạn 3: train baseline (Setup A — image-level split có sẵn)
!python src/train.py --config configs/default.yaml --split_mode image_level

# Giai đoạn 4: kiểm thử logic patient-level split (pseudo-patient trên data tạm)
!python src/train.py --config configs/default.yaml --split_mode pseudo_patient_level

# Đánh giá
!python src/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt

# Giai đoạn 5: demo Grad-CAM
!python src/explain.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --method gradcam --num_samples 8
```

Hoặc mở trực tiếp các notebook trong `notebooks/` — mỗi notebook tương ứng
1 giai đoạn trong kế hoạch thực hiện.

## Roadmap khi có dữ liệu thật

Xem `docs/data_contract.md` để biết chính xác schema/metadata cần bệnh viện
cung cấp (patient_id, ROI annotation, v.v.) để pipeline chạy được ở chế độ
"chính thức" (Setup B thật, đánh giá XAI định lượng thật).
