"""Train/Val 划分 (85/15)。

关键逻辑:
1. 按文件名基数分组 (100a/100b/100c → 基数 "100")，整组划分防数据泄露
2. 分层采样保证类别比例一致
3. 用符号链接创建 images/{train,val}，节省磁盘
"""

import re
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

# --- 配置 ---
DATASET_ROOT = Path.home() / "datasets/power_line_foreign_object"
IMG_DIR = DATASET_ROOT / "JPEGImages"
LABEL_DIR = DATASET_ROOT / "labels"
VAL_RATIO = 0.15
SEED = 42

CLASS_MAP = {"balloon": 0, "kite": 1, "nest": 2, "trash": 3}


def get_group_key(stem: str) -> str:
    """提取文件名基数，用于分组。

    100a → "100", 100b → "100", kuochong_FZ1 → "kuochong_FZ1"
    """
    # 匹配: 数字 + 末尾单个小写字母
    m = re.match(r"^(\d+)[a-z]$", stem)
    if m:
        return m.group(1)
    return stem


def get_dominant_class(stems: list[str], label_dir: Path) -> int:
    """获取一组文件的主要类别 (用于分层采样)。"""
    class_counts = defaultdict(int)
    for stem in stems:
        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
    if not class_counts:
        return -1  # 无标注
    return max(class_counts, key=class_counts.get)


def make_symlinks(stems: list[str], split_name: str):
    """为指定 split 创建图片和标签的符号链接。"""
    img_out = DATASET_ROOT / "images" / split_name
    lbl_out = DATASET_ROOT / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        # 图片符号链接
        img_src = IMG_DIR / f"{stem}.jpg"
        img_dst = img_out / f"{stem}.jpg"
        if img_src.exists() and not img_dst.exists():
            img_dst.symlink_to(img_src)

        # 标签: 复制到子目录 (或符号链接)
        lbl_src = LABEL_DIR / f"{stem}.txt"
        lbl_dst = lbl_out / f"{stem}.txt"
        if lbl_src.exists() and not lbl_dst.exists():
            lbl_dst.symlink_to(lbl_src)


def main():
    # 1. 收集所有 label 文件的 stem
    all_stems = sorted([p.stem for p in LABEL_DIR.glob("*.txt")])
    print(f"共 {len(all_stems)} 个标注文件")

    # 2. 按基数分组
    groups = defaultdict(list)
    for stem in all_stems:
        key = get_group_key(stem)
        groups[key].append(stem)

    group_keys = sorted(groups.keys())
    print(f"分组后 {len(group_keys)} 个独立组")

    # 3. 获取每组的主要类别 (用于分层)
    group_classes = []
    for key in group_keys:
        cls = get_dominant_class(groups[key], LABEL_DIR)
        group_classes.append(cls)

    # 4. 分层划分
    train_keys, val_keys = train_test_split(
        group_keys,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=group_classes,
    )

    train_stems = [s for k in train_keys for s in groups[k]]
    val_stems = [s for k in val_keys for s in groups[k]]

    print(f"Train: {len(train_stems)} 张, Val: {len(val_stems)} 张")
    print(f"比例: {len(val_stems)/len(all_stems)*100:.1f}% val")

    # 5. 统计各集合类别分布
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        class_counts = defaultdict(int)
        for stem in stems:
            lbl = LABEL_DIR / f"{stem}.txt"
            if not lbl.exists():
                continue
            for line in lbl.read_text().strip().split("\n"):
                if line.strip():
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
        id2name = {v: k for k, v in CLASS_MAP.items()}
        dist = {id2name.get(k, k): v for k, v in sorted(class_counts.items())}
        print(f"  {split_name}: {dist}")

    # 6. 创建符号链接
    make_symlinks(train_stems, "train")
    make_symlinks(val_stems, "val")
    print("符号链接创建完成")


if __name__ == "__main__":
    main()
