"""
Quét toàn bộ kết quả đã có trong outputs/ (từ evaluate.py, error_analysis.py,
run_ablation.py) và sinh ra 1 file Markdown chứa các bảng đã điền sẵn số liệu,
đúng cấu trúc bảng đã thiết kế trong báo cáo Chương 5 — để copy trực tiếp vào
báo cáo thay vì gõ tay từng số.

Chỉ tổng hợp những gì ĐÃ CHẠY — phần nào chưa có file kết quả tương ứng sẽ giữ
nguyên placeholder [ĐIỀN SAU KHI CHẠY] kèm gợi ý lệnh cần chạy.

Ví dụ chạy (sau khi đã chạy train/evaluate/error_analysis/ablation ở mức cần thiết):
    !python src/generate_report_tables.py --config configs/default.yaml
"""
import argparse
import json
import os

import pandas as pd

from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả thành bảng Markdown cho báo cáo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/figures/report_tables.md")
    return parser.parse_args()


def try_load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def try_load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def section_5_1_1(report_dir_base: str, backbones: list[str]) -> str:
    """Bảng so sánh backbone — mục 5.1.1. Cần chạy evaluate.py riêng cho từng
    backbone với report_dir khác nhau; script này quét theo quy ước đặt tên
    outputs/figures/backbone_<name>/test_metrics.json (bạn tự đặt report_dir
    tương ứng khi train từng backbone, hoặc dùng run_ablation.py làm mẫu để
    viết thêm 1 runner tương tự cho backbone nếu muốn tự động hoá đầy đủ)."""
    rows = []
    for backbone in backbones:
        metrics = try_load_json(f"outputs/figures/backbone_{backbone}/test_metrics.json")
        if metrics:
            rows.append({
                "Backbone": backbone,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "F1-macro": f"{metrics['f1_macro']:.4f}",
                "Kappa": f"{metrics['kappa_quadratic']:.4f}",
            })
        else:
            rows.append({"Backbone": backbone, "Accuracy": "[ ]", "F1-macro": "[ ]", "Kappa": "[ ]"})

    df = pd.DataFrame(rows)
    md = "### 5.1.1. So sánh backbone (Setup A — image_level)\n\n"
    md += df.to_markdown(index=False)
    if any(r["Accuracy"] == "[ ]" for r in rows):
        md += ("\n\n> ⚠️ Một số backbone chưa có kết quả. Chạy: "
               "`train.py` + `evaluate.py` với `evaluate.report_dir` "
               "= `outputs/figures/backbone_<tên>` cho từng backbone.\n")
    return md


def section_5_1_2() -> str:
    """Bảng so sánh split_mode — mục 5.1.2, đọc trực tiếp 2 file test_metrics.json
    mặc định (evaluate.py ghi đè cùng report_dir mỗi lần chạy — khuyến nghị đổi
    report_dir mỗi split_mode như run_ablation.py làm, để giữ cả 2 kết quả)."""
    setup_a = try_load_json("outputs/figures/split_image_level/test_metrics.json")
    setup_b = try_load_json("outputs/figures/split_pseudo_patient_level/test_metrics.json")

    def fmt(m, key):
        return f"{m[key]:.4f}" if m else "[ ]"

    md = "### 5.1.2. So sánh chiến lược chia dữ liệu (RQ1a — pseudo trên proxy)\n\n"
    md += "| Chiến lược | Accuracy | F1-macro | Kappa |\n|---|---|---|---|\n"
    md += f"| Setup A — image_level | {fmt(setup_a,'accuracy')} | {fmt(setup_a,'f1_macro')} | {fmt(setup_a,'kappa_quadratic')} |\n"
    md += f"| Setup B (pseudo) — pseudo_patient_level | {fmt(setup_b,'accuracy')} | {fmt(setup_b,'f1_macro')} | {fmt(setup_b,'kappa_quadratic')} |\n"

    if setup_a and setup_b:
        diff = setup_a["kappa_quadratic"] - setup_b["kappa_quadratic"]
        md += f"\nChênh lệch Kappa (A - B): **{diff:+.4f}**\n"
        md += ("\n> ⚠️ Nhắc lại: Setup B dùng pseudo-patient GIẢ LẬP trên proxy data — "
               "con số này chỉ xác nhận pipeline chạy đúng, KHÔNG kết luận khoa học về RQ1a thật.\n")
    else:
        md += ("\n> ⚠️ Chưa đủ dữ liệu. Chạy lần lượt:\n"
               "> ```\n"
               "> python src/train.py --config configs/default.yaml --split_mode image_level\n"
               "> # sửa evaluate.report_dir = outputs/figures/split_image_level rồi evaluate.py\n"
               "> python src/train.py --config configs/default.yaml --split_mode pseudo_patient_level\n"
               "> # sửa evaluate.report_dir = outputs/figures/split_pseudo_patient_level rồi evaluate.py\n"
               "> ```\n")
    return md


def section_5_3_1(error_dir: str, class_names: list[str]) -> str:
    """Bảng precision/recall/F1 theo lớp — đọc từ all_predictions.csv của error_analysis.py."""
    df = try_load_csv(os.path.join(error_dir, "all_predictions.csv"))
    md = "### 5.3.1. Precision/Recall/F1 theo lớp\n\n"

    if df is None:
        md += ("\n> ⚠️ Chưa có dữ liệu. Chạy: `python src/error_analysis.py --config configs/default.yaml "
               "--checkpoint outputs/checkpoints/best.pt`\n")
        return md

    from sklearn.metrics import classification_report
    report = classification_report(
        df["y_true"], df["y_pred"], target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cn in class_names:
        r = report.get(cn, {})
        rows.append({
            "Lớp": cn,
            "Precision": f"{r.get('precision', 0):.3f}",
            "Recall": f"{r.get('recall', 0):.3f}",
            "F1": f"{r.get('f1-score', 0):.3f}",
            "Số ảnh test": int(r.get("support", 0)),
        })
    md += pd.DataFrame(rows).to_markdown(index=False)
    return md


def section_5_3_severe_errors(error_dir: str) -> str:
    """Thống kê nhanh lỗi nghiêm trọng (gap >= 3) — bổ sung cho mục 5.3.2/5.3.4."""
    df = try_load_csv(os.path.join(error_dir, "all_predictions.csv"))
    md = "### 5.3.2 (bổ sung). Thống kê lỗi theo độ lệch ordinal\n\n"
    if df is None:
        md += "\n> ⚠️ Chưa có dữ liệu — chạy `error_analysis.py` như trên.\n"
        return md

    n_total = len(df)
    error_df = df[df["y_true"] != df["y_pred"]]
    n_errors = len(error_df)
    n_severe = len(error_df[error_df["gap"] >= 3])

    md += f"- Tổng số ảnh test: **{n_total}**\n"
    md += f"- Tổng số lỗi: **{n_errors}** ({100*n_errors/n_total:.1f}%)\n"
    md += f"- Lỗi nghiêm trọng (lệch ≥ 3 mức): **{n_severe}** ({100*n_severe/max(n_errors,1):.1f}% trong số lỗi)\n"
    md += "\nXem `error_grid.png` cùng thư mục để xem trực tiếp các ảnh lỗi nghiêm trọng nhất.\n"
    return md


def section_5_4(ablation_csv: str) -> str:
    df = try_load_csv(ablation_csv)
    md = "### 5.4. Ablation Study — kết quả đã chạy được trên proxy data\n\n"
    if df is None:
        md += ("\n> ⚠️ Chưa chạy ablation. Chạy: `python src/run_ablation.py "
               "--config configs/default.yaml --ablation both`\n")
        return md

    display_cols = [c for c in ["variant_name", "accuracy", "f1_macro", "kappa_quadratic"] if c in df.columns]
    md += df[display_cols].to_markdown(index=False)
    return md


def main():
    args = parse_args()
    cfg = load_config(args.config)
    class_names = cfg["data"]["class_names"]
    error_dir = os.path.join(cfg["evaluate"]["report_dir"], "error_analysis")

    sections = [
        "# Bảng kết quả tổng hợp — sinh tự động từ outputs/\n",
        "*(File này được sinh bởi `src/generate_report_tables.py`. Copy các bảng "
        "đã điền số liệu vào đúng vị trí tương ứng trong báo cáo Chương 5.)*\n",
        section_5_1_1("outputs/figures", backbones=["mobilenet_v2", "resnet18", "efficientnet_b0"]),
        section_5_1_2(),
        section_5_3_1(error_dir, class_names),
        section_5_3_severe_errors(error_dir),
        section_5_4("outputs/figures/ablation_results.csv"),
    ]

    output_content = "\n\n---\n\n".join(sections)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Đã sinh bảng tổng hợp tại: {args.output}")
    print("Mở file này để copy các bảng đã điền số liệu vào báo cáo Chương 5.")


if __name__ == "__main__":
    main()
