# 电力线异物检测 — RK3588 部署

YOLOv5s 模型部署到 RK3588 NPU 的完整工具链。

## 目录结构

```
├── model/
│   ├── best.onnx          # RKNN 优化的 ONNX 模型 (airockchip/yolov5 --rknpu 导出)
│   ├── RK_anchors.txt     # anchor 参数
│   └── best.rknn          # 转换后的 RKNN 模型（板端生成）
├── convert_rknn.py        # ONNX → RKNN 转换脚本（x86 PC 上运行）
├── inference_rk3588.py    # RK3588 板端推理脚本
├── dataset.txt            # INT8 量化校准图片列表
└── calibration/           # 校准图片（100张，不上传 Git）
```

## RK3588 板端使用

### 1. 安装依赖
```bash
pip install rknn-toolkit-lite2 opencv-python numpy
```

### 2. 转换模型（如果还没有 .rknn 文件）
在 x86 PC 上运行（需安装 rknn-toolkit2）：
```bash
pip install rknn-toolkit2
python convert_rknn.py
```

### 3. 推理
```bash
python inference_rk3588.py --img test.jpg --save result.jpg
```

## 模型信息
- 架构: YOLOv5s
- 输入: 640×640 RGB
- 类别: balloon(气球), kite(风筝), nest(鸟巢), trash(垃圾)
- 训练 mAP@0.5: 0.988
