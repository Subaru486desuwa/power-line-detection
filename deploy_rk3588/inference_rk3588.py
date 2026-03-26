"""RK3588 板端推理脚本（需安装 rknn-toolkit-lite2 + opencv-python）。

用法:
    python inference_rk3588.py --img test.jpg
    python inference_rk3588.py --img test.jpg --save result.jpg
"""

import argparse
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ---- 配置 ----
RKNN_MODEL = "model/best.rknn"
IMG_SIZE = 640
CONF_THRESH = 0.50
NMS_THRESH = 0.45
NUM_CLASSES = 4

CLASS_NAMES = ["balloon", "kite", "nest", "trash"]
CLASS_NAMES_CN = {"balloon": "气球", "kite": "风筝", "nest": "鸟巢", "trash": "垃圾"}
CLASS_COLORS = {
    "balloon": (0, 0, 230),
    "kite": (0, 140, 255),
    "nest": (0, 180, 0),
    "trash": (180, 0, 180),
}

# YOLOv5 默认 anchors（从 RK_anchors.txt）
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],     # P3/8
    [[30, 61], [62, 45], [59, 119]],     # P4/16
    [[116, 90], [156, 198], [373, 326]], # P5/32
]
STRIDES = [8, 16, 32]


def letterbox(img, new_shape=(640, 640)):
    """Letterbox resize，保持宽高比，灰边填充。"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_outputs(outputs):
    """将 3 个原始特征图解码为 [x1, y1, x2, y2, conf, cls_id] 格式。"""
    all_boxes = []

    for i, feat in enumerate(outputs):
        # feat shape: (1, 27, H, W) -> (3, 9, H, W)
        if feat.ndim == 4:
            feat = feat[0]
        na = 3
        no = 5 + NUM_CLASSES
        h, w = feat.shape[1], feat.shape[2]
        feat = feat.reshape(na, no, h, w).transpose(0, 2, 3, 1)  # (3, H, W, 9)

        stride = STRIDES[i]
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)

        for a in range(na):
            box_data = feat[a]  # (H, W, 9)
            xy = (sigmoid(box_data[..., :2]) * 2 - 0.5 + grid) * stride
            wh = (sigmoid(box_data[..., 2:4]) * 2) ** 2 * np.array(ANCHORS[i][a])
            obj_conf = sigmoid(box_data[..., 4:5])
            cls_conf = sigmoid(box_data[..., 5:])

            conf = obj_conf * cls_conf
            cls_id = np.argmax(conf, axis=-1)
            max_conf = np.max(conf, axis=-1)

            mask = max_conf > CONF_THRESH
            if not mask.any():
                continue

            xy_sel = xy[mask]
            wh_sel = wh[mask]
            conf_sel = max_conf[mask]
            cls_sel = cls_id[mask]

            x1 = xy_sel[:, 0] - wh_sel[:, 0] / 2
            y1 = xy_sel[:, 1] - wh_sel[:, 1] / 2
            x2 = xy_sel[:, 0] + wh_sel[:, 0] / 2
            y2 = xy_sel[:, 1] + wh_sel[:, 1] / 2

            for j in range(len(conf_sel)):
                all_boxes.append([x1[j], y1[j], x2[j], y2[j], conf_sel[j], cls_sel[j]])

    return np.array(all_boxes) if all_boxes else np.zeros((0, 6))


def nms(boxes, iou_thresh=NMS_THRESH):
    """按类别做 NMS。"""
    if len(boxes) == 0:
        return boxes
    result = []
    for cls_id in range(NUM_CLASSES):
        cls_mask = boxes[:, 5] == cls_id
        cls_boxes = boxes[cls_mask]
        if len(cls_boxes) == 0:
            continue
        x1, y1, x2, y2 = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
        scores = cls_boxes[:, 4]
        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=scores.tolist(),
            score_threshold=CONF_THRESH,
            nms_threshold=iou_thresh,
        )
        if len(indices) > 0:
            indices = indices.flatten()
            result.append(cls_boxes[indices])
    return np.vstack(result) if result else np.zeros((0, 6))


def draw_results(img, boxes, ratio, pad):
    """在原始图像上绘制检测框。"""
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1 = int((x1 - pad[0]) / ratio)
        y1 = int((y1 - pad[1]) / ratio)
        x2 = int((x2 - pad[0]) / ratio)
        y2 = int((y2 - pad[1]) / ratio)
        cls_name = CLASS_NAMES[int(cls_id)]
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES_CN.get(cls_name, cls_name)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="输入图片路径")
    parser.add_argument("--model", default=RKNN_MODEL, help="RKNN 模型路径")
    parser.add_argument("--save", default=None, help="输出图片保存路径")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    args = parser.parse_args()

    global CONF_THRESH
    CONF_THRESH = args.conf

    # 初始化 RKNN
    rknn = RKNNLite(verbose=False)
    print(f"Loading model: {args.model}")
    ret = rknn.load_rknn(args.model)
    assert ret == 0, "Load RKNN model failed!"

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)  # 三核 NPU
    assert ret == 0, "Init runtime failed!"

    # 读图 + 预处理
    img_orig = cv2.imread(args.img)
    assert img_orig is not None, f"Cannot read image: {args.img}"
    img_lb, ratio, pad = letterbox(img_orig, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

    # 推理
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[np.expand_dims(img_rgb, 0)])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 后处理
    boxes = decode_outputs(outputs)
    boxes = nms(boxes)

    # 统计
    stats = {n: 0 for n in CLASS_NAMES}
    for box in boxes:
        stats[CLASS_NAMES[int(box[5])]] += 1

    print(f"Inference: {elapsed_ms:.1f}ms")
    print(f"Detections: {stats}")
    print(f"Total: {len(boxes)}")

    # 画框
    result_img = draw_results(img_orig.copy(), boxes, ratio, pad)
    if args.save:
        cv2.imwrite(args.save, result_img)
        print(f"Saved to {args.save}")
    else:
        cv2.imshow("Detection", result_img)
        cv2.waitKey(0)

    rknn.release()


if __name__ == "__main__":
    main()
