"""RKNN 推理封装：加载 RKNN 模型、检测、画框（RK3588 板端使用）。

与 detector.py (PyTorch) 保持相同的公共接口：
    - detect(frame_bgr) -> (annotated, stats, elapsed_ms)
    - set_conf(conf)
    - device_name
"""

import time
from collections import Counter

import cv2
import numpy as np

from resources import CLASS_COLORS, CLASS_NAMES_CN, RKNN_MODEL_PATH

# 延迟导入，仅在 RK3588 上可用
from rknnlite.api import RKNNLite

# ---- 常量 ----
IMG_SIZE = 640
NUM_CLASSES = 4
NMS_THRESH = 0.45
CLASS_NAMES = ["balloon", "kite", "nest", "trash"]

ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
]
STRIDES = [8, 16, 32]


def _letterbox(img, new_shape=(640, 640)):
    """Letterbox resize，保持宽高比。"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def _decode_outputs(outputs, conf_thresh):
    """将 3 个原始特征图解码为 [x1, y1, x2, y2, conf, cls_id]。"""
    all_boxes = []
    for i, feat in enumerate(outputs):
        if feat.ndim == 4:
            feat = feat[0]
        na, no = 3, 5 + NUM_CLASSES
        h, w = feat.shape[1], feat.shape[2]
        feat = feat.reshape(na, no, h, w).transpose(0, 2, 3, 1)

        stride = STRIDES[i]
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        for a in range(na):
            d = feat[a]
            xy = (_sigmoid(d[..., :2]) * 2 - 0.5 + grid) * stride
            wh = (_sigmoid(d[..., 2:4]) * 2) ** 2 * np.array(ANCHORS[i][a])
            obj = _sigmoid(d[..., 4:5])
            cls = _sigmoid(d[..., 5:])
            conf = obj * cls
            cls_id = np.argmax(conf, axis=-1)
            max_conf = np.max(conf, axis=-1)

            mask = max_conf > conf_thresh
            if not mask.any():
                continue

            xy_s, wh_s = xy[mask], wh[mask]
            x1 = xy_s[:, 0] - wh_s[:, 0] / 2
            y1 = xy_s[:, 1] - wh_s[:, 1] / 2
            x2 = xy_s[:, 0] + wh_s[:, 0] / 2
            y2 = xy_s[:, 1] + wh_s[:, 1] / 2
            for j in range(len(x1)):
                all_boxes.append([x1[j], y1[j], x2[j], y2[j],
                                  max_conf[mask][j], cls_id[mask][j]])

    return np.array(all_boxes) if all_boxes else np.zeros((0, 6))


def _nms(boxes, conf_thresh, iou_thresh=NMS_THRESH):
    """按类别 NMS。"""
    if len(boxes) == 0:
        return boxes
    result = []
    for c in range(NUM_CLASSES):
        m = boxes[:, 5] == c
        cb = boxes[m]
        if len(cb) == 0:
            continue
        x1, y1, x2, y2 = cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3]
        indices = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            cb[:, 4].tolist(), conf_thresh, iou_thresh,
        )
        if len(indices) > 0:
            result.append(cb[indices.flatten()])
    return np.vstack(result) if result else np.zeros((0, 6))


class DetectorRKNN:
    """RKNN 推理封装，接口与 Detector (PyTorch) 一致。"""

    def __init__(self, conf: float = 0.50, model_path: str = None):
        self._conf = conf
        path = model_path or RKNN_MODEL_PATH
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(path)
        assert ret == 0, f"Load RKNN model failed: {path}"
        # 尝试三核，失败回退单核
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            assert ret == 0, "Init RKNN runtime failed!"
        self._model_path = path

    def set_conf(self, conf: float) -> None:
        self._conf = conf

    def detect(self, frame_bgr: np.ndarray):
        """与 Detector.detect() 相同的接口。"""
        h0, w0 = frame_bgr.shape[:2]
        img_lb, ratio, pad = _letterbox(frame_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = self._rknn.inference(inputs=[np.expand_dims(img_rgb, 0)])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = _decode_outputs(outputs, self._conf)
        boxes = _nms(boxes, self._conf)

        annotated = frame_bgr.copy()
        counter = Counter()

        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            # 坐标还原到原图
            x1 = (x1 - pad[0]) / ratio
            y1 = (y1 - pad[1]) / ratio
            x2 = (x2 - pad[0]) / ratio
            y2 = (y2 - pad[1]) / ratio
            xyxy = [x1, y1, x2, y2]
            name = CLASS_NAMES[int(cls_id)]
            counter[name] += 1
            self._draw_box(annotated, xyxy, conf, name)

        stats = {n: counter.get(n, 0) for n in CLASS_COLORS}
        return annotated, stats, elapsed_ms

    @property
    def device_name(self) -> str:
        return "RK3588 NPU"

    @staticmethod
    def _draw_box(img, xyxy, conf, name):
        """绘制检测框（cv2 版本，兼容无中文字体环境）。"""
        x1, y1, x2, y2 = map(int, xyxy)
        color = CLASS_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES_CN.get(name, name)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = max(y1 - th - 6, 0)
        cv2.rectangle(img, (x1, label_y), (x1 + tw + 4, label_y + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def release(self):
        self._rknn.release()
