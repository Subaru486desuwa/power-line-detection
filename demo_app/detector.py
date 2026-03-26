"""YOLOv5 推理封装：加载模型、检测、画框。"""

import time
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image

from resources import YOLOV5_DIR, WEIGHTS_PATH, CLASS_COLORS, CLASS_NAMES_CN


class Detector:
    """封装 YOLOv5 模型加载与推理。"""

    def __init__(self, conf: float = 0.50):
        self.model = torch.hub.load(
            YOLOV5_DIR, "custom", path=WEIGHTS_PATH,
            source="local", device="0",
        )
        self.model.conf = conf
        self.class_names: dict[int, str] = self.model.names  # {0:'balloon',...}

    # ------ 公共接口 ------

    def set_conf(self, conf: float) -> None:
        """动态调整置信度阈值。"""
        self.model.conf = conf

    def detect(self, frame_bgr: np.ndarray):
        """
        对 BGR 帧执行检测。

        Returns:
            annotated: 带检测框的 BGR 帧
            stats: {'balloon': 2, 'kite': 1, ...} 各类别计数
            elapsed_ms: 推理耗时(ms)
        """
        t0 = time.perf_counter()
        results = self.model(frame_bgr[..., ::-1])  # BGR→RGB 送入模型
        elapsed_ms = (time.perf_counter() - t0) * 1000

        det = results.xyxy[0].cpu().numpy()  # (N, 6): x1,y1,x2,y2,conf,cls
        annotated = frame_bgr.copy()
        counter: Counter = Counter()

        for *xyxy, conf, cls_id in det:
            name = self.class_names[int(cls_id)]
            counter[name] += 1
            self._draw_box(annotated, xyxy, conf, name)

        # 保证所有类别都在统计中
        stats = {n: counter.get(n, 0) for n in CLASS_COLORS}
        return annotated, stats, elapsed_ms

    @property
    def device_name(self) -> str:
        """返回当前设备名称，用于状态栏显示。"""
        dev = next(self.model.model.parameters()).device
        if dev.type == "cuda":
            return torch.cuda.get_device_name(dev)
        return "CPU"

    # ------ 内部方法 ------

    # 中文字体（类级别缓存，只加载一次）
    _font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20)

    @staticmethod
    def _draw_box(img, xyxy, conf, name):
        """在图像上绘制检测框和中文标签。"""
        x1, y1, x2, y2 = map(int, xyxy)
        color = CLASS_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES_CN.get(name, name)} {conf:.2f}"

        # PIL 绘制中文
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        bbox = draw.textbbox((0, 0), label, font=Detector._font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 标签背景
        label_y = max(y1 - th - 8, 0)
        color_rgb = color[::-1]  # BGR→RGB
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=color_rgb)
        draw.text((x1 + 4, label_y + 1), label, fill=(255, 255, 255), font=Detector._font)
        # 写回 numpy
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
