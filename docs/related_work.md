# Paper liên quan theo từng phần của đề án

## 1. Nền tảng lâm sàng — Thang đo BITSS

| Paper | Vai trò |
|---|---|
| Vandenplas et al. (2017), *Development of the Brussels Infant and Toddler Stool Scale ("BITSS"): Protocol of the study*, BMJ Open | Paper gốc phát triển thang BITSS |
| Huysentruyt et al. (2019), *The Brussels Infant and Toddler Stool Scale: A Study on Interobserver Reliability* | Độ tin cậy liên quan sát viên của BITSS — cơ sở so sánh với độ tin cậy gán nhãn trong đồ án |
| Hofman et al. (2022), *Intra-rater Variability of the BITSS Using Photographed Stools*, JPGN | Biến thiên tự đánh giá lại của con người — đặt kỳ vọng hợp lý cho hiệu năng model |
| Bui et al. (2024), *BITSS for hard stool — South Asian perspective*, JGH | Khác biệt vùng miền khi áp dụng BITSS — liên hệ generalization theo site |

## 2. Hai bài báo nền — phân tích gap

| Paper | Vai trò |
|---|---|
| Ludwig et al. (2021), *Machine Learning Supports Automated Digital Image Scoring of Stool Consistency in Diapers*, JPGN 72(2):255-261 | 2687 ảnh/96 trẻ, MobileNet transfer learning — đối tượng phân tích cho RQ1 |
| Xiao et al. (2023), *Generation and application of a CNN algorithm in evaluating stool consistency in diapers*, Acta Paediatrica 112(6):1333-1340 | Không có interpretability — đối tượng phân tích cho RQ2 |
| Wang et al., *The Effectiveness of AI in Assisting Mothers... Breastfeeding Cohort Study in China*, Nutrients 16(6):855 (2024) | Ứng dụng thực tế tiếp theo — minh họa hệ quả nếu shortcut không bị phát hiện trước khi triển khai |

## 3. Shortcut Learning & Data Leakage — nền lý thuyết RQ1

| Paper | Vai trò |
|---|---|
| Geirhos et al. (2020), *Shortcut learning in deep neural networks*, Nature Machine Intelligence 2:665–673 | Định nghĩa khái niệm shortcut learning |
| Hill, Koback & Schilling (2024), *The risk of shortcutting in deep learning algorithms for medical imaging research*, Scientific Reports | Case study CNN "dự đoán" việc ăn đậu/uống bia từ X-quang đầu gối |
| Banerjee et al. (2023), *"Shortcuts" Causing Bias in Radiology AI: Causes, Evaluation, and Mitigation*, JACR | Khung phân loại nguyên nhân/đánh giá/khắc phục shortcut |
| Tampu, Eklund & Haj-Hosseini (2022), *Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images*, Scientific Data | So sánh per-image vs per-subject split, định lượng inflation — mẫu phương pháp gần nhất cho RQ1a |
| "How You Split Matters" (2023), brain MRI, arXiv 2309.00350 | Kết hợp patient-level split + Grad-CAM phát hiện identity confounding |
| PMC9533091, *Mitigating Bias in Radiology Machine Learning: 1. Data Handling* | Hướng dẫn thực hành patient-level splitting |

## 4. XAI cho ảnh y tế — nền lý thuyết RQ2

| Paper | Vai trò |
|---|---|
| Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV | Paper gốc Grad-CAM |
| Panboonyuen (2026), *Seeing Isn't Always Believing: Analysis of Grad-CAM Faithfulness and Localization Reliability in Lung Cancer CT Classification* | Khung đánh giá định lượng faithfulness của Grad-CAM |
| Sattarzadeh et al. (Shap-CAM), arXiv 2208.03608 | Công thức Average Drop / Average Increase in Confidence (dùng trong `src/metrics_xai.py`) |
| Review (2025), *Beyond Post hoc Explanations: A Comprehensive Framework for Accountable AI in Medical Imaging*, PMC12383817 | Meta-analysis 67 nghiên cứu — cơ sở chọn Grad-CAM/LIME ưu tiên hơn SHAP trên ảnh |
| PLOS ONE (2024), *Evaluating XAI techniques in chest radiology imaging through a human-centered Lens* | Khuyến nghị Grad-CAM + LIME, bỏ SHAP với dữ liệu ảnh |
| Boland et al. (2024–2025), *Preventing Shortcut Learning in Medical Image Analysis through Intermediate Layer Knowledge Distillation*, MELBA | Hướng khắc phục shortcut — hướng mở rộng nếu có thời gian |
