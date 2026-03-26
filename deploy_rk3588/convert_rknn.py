"""ONNX → RKNN 转换脚本（在 x86 PC 上运行，需安装 rknn-toolkit2）。"""

import sys
from rknn.api import RKNN

# ---- 配置 ----
ONNX_MODEL = "model/best.onnx"
RKNN_MODEL = "model/best.rknn"
DATASET_TXT = "dataset.txt"
TARGET_PLATFORM = "rk3588"
QUANTIZE = True  # True=INT8 量化（推荐），False=FP16


def main():
    rknn = RKNN(verbose=True)

    # 1. 配置模型
    print("--> Configuring model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET_PLATFORM,
    )

    # 2. 加载 ONNX
    print("--> Loading ONNX model")
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print("Load ONNX model failed!")
        sys.exit(1)

    # 3. 构建（量化或不量化）
    print(f"--> Building RKNN model (quantize={QUANTIZE})")
    ret = rknn.build(do_quantization=QUANTIZE, dataset=DATASET_TXT)
    if ret != 0:
        print("Build RKNN model failed!")
        sys.exit(1)

    # 4. 导出
    print(f"--> Exporting to {RKNN_MODEL}")
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print("Export RKNN model failed!")
        sys.exit(1)

    rknn.release()
    print(f"Done! RKNN model saved to {RKNN_MODEL}")


if __name__ == "__main__":
    main()
