"""常量定义：路径、颜色、中文类名映射。"""

from pathlib import Path

# 路径
_PROJECT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = str(_PROJECT / "yolov5")
WEIGHTS_PATH = str(_PROJECT / "runs" / "power_line" / "weights" / "best.pt")

# BGR 颜色（与 cv2 一致）
CLASS_COLORS = {
    "balloon": (0, 0, 230),    # 红
    "kite":    (0, 140, 255),  # 橙
    "nest":    (0, 180, 0),    # 绿
    "trash":   (180, 0, 180),  # 紫
}

# 中文类名
CLASS_NAMES_CN = {
    "balloon": "气球",
    "kite":    "风筝",
    "nest":    "鸟巢",
    "trash":   "垃圾",
}
