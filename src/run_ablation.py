"""
Chạy tự động các thí nghiệm ablation đã xác định là KHẢ THI trên dataset proxy
(Chương 5.4 báo cáo — 2 ablation đầu tiên, không cần dữ liệu thật/ROI):

    1. Freeze backbone: toàn bộ / 2 block cuối / toàn bộ mở (fine-tune hết)
    2. Augmentation: bật / tắt

Mỗi cấu hình được train từ đầu (gọi lại train.py's main logic qua subprocess
để đảm bảo mỗi lần chạy độc lập, không rò rỉ trạng thái global giữa các lần),
sau đó evaluate.py được gọi để lấy metric, và toàn bộ kết quả được tổng hợp
thành 1 bảng CSV duy nhất — dùng trực tiếp cho bảng 5.4 trong báo cáo.

⚠️ Lưu ý: mỗi cấu hình train từ đầu, tốn thời gian gấp N lần so với train 1 lần.
Trên Colab free tier, cân nhắc giảm cfg['train']['epochs'] tạm thời khi chạy
ablation để không vượt quá giới hạn session.

Ví dụ chạy:
    !python src/run_ablation.py --config configs/default.yaml --ablation freeze
    !python src/run_ablation.py --config configs/default.yaml --ablation augmentation
    !python src/run_ablation.py --config configs/default.yaml --ablation both
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy ablation study tự động trên proxy data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ablation", type=str, default="both",
                         choices=["freeze", "augmentation", "both"],
                         help="Loại ablation muốn chạy")
    parser.add_argument("--output_csv", type=str, default="outputs/figures/ablation_results.csv")
    return parser.parse_args()


def get_freeze_variants(base_cfg: dict) -> list[dict]:
    """3 biến thể chiến lược fine-tuning — chỉ cần đổi model.freeze_backbone /
    unfreeze_last_n_blocks, không cần dữ liệu bổ sung."""
    variants = []

    cfg_frozen = copy.deepcopy(base_cfg)
    cfg_frozen["model"]["freeze_backbone"] = True
    cfg_frozen["model"]["unfreeze_last_n_blocks"] = 0
    variants.append(("freeze_all", cfg_frozen))

    cfg_partial = copy.deepcopy(base_cfg)
    cfg_partial["model"]["freeze_backbone"] = True
    cfg_partial["model"]["unfreeze_last_n_blocks"] = 2
    variants.append(("freeze_partial_2blocks", cfg_partial))

    cfg_full = copy.deepcopy(base_cfg)
    cfg_full["model"]["freeze_backbone"] = False
    variants.append(("fine_tune_full", cfg_full))

    return variants


def get_augmentation_variants(base_cfg: dict) -> list[dict]:
    """2 biến thể augmentation on/off."""
    variants = []

    cfg_aug_on = copy.deepcopy(base_cfg)
    variants.append(("augmentation_on", cfg_aug_on))  # giữ nguyên config gốc

    cfg_aug_off = copy.deepcopy(base_cfg)
    cfg_aug_off["augmentation"]["train"] = {
        "horizontal_flip": False, "rotation_degrees": 0,
        "color_jitter_brightness": 0, "color_jitter_contrast": 0,
    }
    variants.append(("augmentation_off", cfg_aug_off))

    return variants


def run_one_variant(name: str, cfg: dict, repo_root: str) -> dict:
    """Ghi config biến thể ra file tạm, gọi train.py + evaluate.py bằng subprocess,
    đọc lại test_metrics.json, dọn dẹp file tạm.
    """
    print(f"\n{'='*60}\nĐang chạy biến thể: {name}\n{'='*60}")

    # Mỗi biến thể dùng checkpoint/report riêng để không ghi đè lẫn nhau
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["train"]["checkpoint_dir"] = f"outputs/checkpoints/ablation_{name}"
    variant_cfg["evaluate"]["checkpoint"] = f"outputs/checkpoints/ablation_{name}/best.pt"
    variant_cfg["evaluate"]["report_dir"] = f"outputs/figures/ablation_{name}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                       dir=os.path.join(repo_root, "configs")) as tmp_f:
        yaml.safe_dump(variant_cfg, tmp_f, allow_unicode=True)
        tmp_config_path = tmp_f.name

    try:
        # Dùng cùng python executable đang chạy script này, đảm bảo đúng môi trường (Colab-safe)
        subprocess.run(
            [sys.executable, os.path.join(repo_root, "src", "train.py"), "--config", tmp_config_path],
            check=True, cwd=repo_root,
        )
        subprocess.run(
            [sys.executable, os.path.join(repo_root, "src", "evaluate.py"), "--config", tmp_config_path],
            check=True, cwd=repo_root,
        )

        metrics_path = os.path.join(repo_root, variant_cfg["evaluate"]["report_dir"], "test_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["variant_name"] = name
        return metrics

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Biến thể '{name}' THẤT BẠI (lỗi subprocess): {e}")
        return {"variant_name": name, "error": str(e)}
    finally:
        # data_paths.yaml nằm cùng thư mục configs/ nên tempfile ở đó vẫn merge đúng
        os.remove(tmp_config_path)


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    # Cách xác định repo_root ổn định hơn: dùng vị trí script này
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    variants = []
    if args.ablation in ("freeze", "both"):
        variants += get_freeze_variants(base_cfg)
    if args.ablation in ("augmentation", "both"):
        variants += get_augmentation_variants(base_cfg)

    print(f"Sẽ chạy tổng cộng {len(variants)} biến thể: {[n for n, _ in variants]}")

    results = []
    for name, variant_cfg in variants:
        result = run_one_variant(name, variant_cfg, repo_root)
        results.append(result)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\n{'='*60}\nHOÀN TẤT ABLATION STUDY\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nĐã lưu bảng tổng hợp vào {args.output_csv} — dùng trực tiếp cho Chương 5.4 báo cáo.")


if __name__ == "__main__":
    main()
