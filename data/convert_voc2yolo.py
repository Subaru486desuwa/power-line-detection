"""VOC XML → YOLO TXT 格式转换。

类别映射: balloon=0, kite=1, nest=2, trash=3
输出: ~/datasets/power_line_foreign_object/labels/<stem>.txt
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# --- 配置 ---
ANNO_DIR = Path.home() / "datasets/power_line_foreign_object/Annotations"
LABEL_DIR = Path.home() / "datasets/power_line_foreign_object/labels"
CLASS_MAP = {"balloon": 0, "kite": 1, "nest": 2, "trash": 3}


def convert_one(xml_path: Path, out_dir: Path) -> int:
    """转换单个 XML，返回目标数量。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    if w == 0 or h == 0:
        print(f"[WARN] {xml_path.name}: width/height 为 0，跳过")
        return 0

    lines = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            print(f"[WARN] {xml_path.name}: 未知类别 '{cls_name}'，跳过")
            continue

        cls_id = CLASS_MAP[cls_name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # 归一化 (cx, cy, w, h)
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        # 裁剪到 [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    out_path = out_dir / f"{xml_path.stem}.txt"
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ANNO_DIR.glob("*.xml"))
    print(f"共 {len(xml_files)} 个 XML 文件")

    total_objs = 0
    for xf in xml_files:
        total_objs += convert_one(xf, LABEL_DIR)

    txt_count = len(list(LABEL_DIR.glob("*.txt")))
    print(f"转换完成: {txt_count} 个 label 文件, 共 {total_objs} 个目标")


if __name__ == "__main__":
    main()
