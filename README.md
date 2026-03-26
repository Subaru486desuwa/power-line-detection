# 电力线路异物检测系统

基于 YOLOv5s 的电力线路异物（气球、风筝、鸟巢、垃圾）检测系统。

## 项目结构

```
├── data/                    # 数据集配置和转换脚本
├── demo_app/                # PySide6 桌面演示应用（PC 端）
├── deploy_rk3588/           # RK3588 NPU 部署工具链
│   ├── model/               # ONNX 模型 + anchors
│   ├── convert_rknn.py      # ONNX → RKNN 转换
│   └── inference_rk3588.py  # 板端推理脚本
├── runs/power_line/weights/ # 训练权重
└── train.py                 # 训练脚本
```

## 模型指标
- 架构: YOLOv5s (7M params)
- mAP@0.5: 0.988
- 类别: balloon(气球), kite(风筝), nest(鸟巢), trash(垃圾)

## PC 端演示
```bash
cd demo_app && python main.py
```

## RK3588 部署
详见 [deploy_rk3588/README.md](deploy_rk3588/README.md)
