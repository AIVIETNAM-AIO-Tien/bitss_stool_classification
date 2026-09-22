"""
Unit test nhanh cho src/dataset.py — chạy được ngay cả khi chưa có dữ liệu
thật, dùng ảnh giả (random tensor lưu thành .jpg) để kiểm tra LOGIC, không
kiểm tra nội dung ảnh.

Chạy: python -m pytest tests/test_dataset.py -v
(hoặc python tests/test_dataset.py nếu không có pytest trên Colab)
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import (  # noqa: E402
    _list_image_folder_samples,
    _assign_pseudo_patient_ids,
    _group_split,
    class_distribution_report,
)


def _make_fake_dataset(root_dir: str, class_names: list[str], n_per_class: int = 6):
    for split in ["train", "valid", "test"]:
        for class_name in class_names:
            class_dir = os.path.join(root_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(n_per_class):
                arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(class_dir, f"img_{i}.jpg"))


def test_list_image_folder_samples_reads_all_classes():
    with tempfile.TemporaryDirectory() as tmp:
        class_names = [f"Type_{i}" for i in range(1, 8)]
        _make_fake_dataset(tmp, class_names, n_per_class=4)

        samples = _list_image_folder_samples(os.path.join(tmp, "train"), class_names)
        assert len(samples) == 7 * 4, "Phải đọc đủ 7 lớp x 4 ảnh"
        labels = {s.label for s in samples}
        assert labels == set(range(7)), "Nhãn phải đánh số 0..6 tương ứng Type_1..7"


def test_pseudo_patient_grouping_no_leakage_after_split():
    with tempfile.TemporaryDirectory() as tmp:
        class_names = [f"Type_{i}" for i in range(1, 8)]
        _make_fake_dataset(tmp, class_names, n_per_class=20)  # đủ ảnh để nhóm 4 ảnh/pseudo-patient

        all_samples = []
        for split in ["train", "valid", "test"]:
            all_samples += _list_image_folder_samples(os.path.join(tmp, split), class_names)

        pseudo_samples = _assign_pseudo_patient_ids(all_samples, images_per_group=4, seed=42)

        # Kiểm tra bất biến quan trọng nhất: sau khi split, không group nào lọt cả 2 tập
        train_samples, test_samples = _group_split(pseudo_samples, test_size=0.2, seed=42)
        train_groups = {s.group_id for s in train_samples}
        test_groups = {s.group_id for s in test_samples}
        assert train_groups.isdisjoint(test_groups), (
            "LỖI: pseudo-patient group bị rò rỉ giữa train/test — "
            "đây chính là lỗi mà patient-level split phải ngăn chặn (RQ1)."
        )


def test_class_distribution_report_shape():
    with tempfile.TemporaryDirectory() as tmp:
        class_names = [f"Type_{i}" for i in range(1, 8)]
        _make_fake_dataset(tmp, class_names, n_per_class=5)

        df = class_distribution_report(tmp, class_names)
        assert len(df) == 3 * 7, "3 splits x 7 classes"
        assert (df["count"] == 5).all(), "Mỗi lớp phải có đúng 5 ảnh giả đã tạo"


if __name__ == "__main__":
    test_list_image_folder_samples_reads_all_classes()
    test_pseudo_patient_grouping_no_leakage_after_split()
    test_class_distribution_report_shape()
    print("Tất cả test PASS.")
