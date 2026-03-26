"""常量定义：路径、颜色、中文类名映射。"""

from pathlib import Path

# 路径
_PROJECT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = str(_PROJECT / "yolov5")
WEIGHTS_PATH = str(_PROJECT / "runs" / "power_line" / "weights" / "best.pt")

# RKNN 模型路径（RK3588 板端使用，优先 i8，回退 fp16）
_rknn_dir = _PROJECT / "deploy_rk3588" / "model"
RKNN_MODEL_PATH = str(
    _rknn_dir / "best_i8.rknn"
    if (_rknn_dir / "best_i8.rknn").exists()
    else _rknn_dir / "best_fp16.rknn"
)

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
