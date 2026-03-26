"""ONNX -> RKNN 转换脚本（在 x86 PC 上运行，需安装 rknn-toolkit2）。

支持多种量化方案，方便对比精度差异：
    python convert_rknn.py                    # 默认 INT8 量化
    python convert_rknn.py --mode fp          # FP16 不量化
    python convert_rknn.py --mode i8          # INT8 量化
    python convert_rknn.py --mode all         # 一次生成全部方案
"""

import argparse
import sys
import os
from rknn.api import RKNN

ONNX_MODEL = "model/best.onnx"
DATASET_TXT = "dataset.txt"
TARGET_PLATFORM = "rk3588"

# 量化方案配置
QUANT_MODES = {
    "fp": {
        "desc": "FP16 (不量化，精度最高，速度较慢)",
        "output": "model/best_fp16.rknn",
        "quantize": False,
    },
    "i8": {
        "desc": "INT8 (量化，速度最快，可能有精度损失)",
        "output": "model/best_i8.rknn",
        "quantize": True,
    },
}


def convert(mode_name: str, cfg: dict) -> bool:
    """执行单次转换，返回是否成功。"""
    print(f"\n{'='*60}")
    print(f"  方案: {mode_name} — {cfg['desc']}")
    print(f"  输出: {cfg['output']}")
    print(f"{'='*60}\n")

    rknn = RKNN(verbose=True)

    # 1. 配置
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET_PLATFORM,
    )

    # 2. 加载 ONNX
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print(f"[{mode_name}] Load ONNX failed!")
        rknn.release()
        return False

    # 3. 构建
    ret = rknn.build(
        do_quantization=cfg["quantize"],
        dataset=DATASET_TXT if cfg["quantize"] else None,
    )
    if ret != 0:
        print(f"[{mode_name}] Build failed!")
        rknn.release()
        return False

    # 4. 导出
    os.makedirs(os.path.dirname(cfg["output"]), exist_ok=True)
    ret = rknn.export_rknn(cfg["output"])
    rknn.release()

    if ret != 0:
        print(f"[{mode_name}] Export failed!")
        return False

    size_mb = os.path.getsize(cfg["output"]) / 1024 / 1024
    print(f"[{mode_name}] Done! {cfg['output']} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX → RKNN 转换（支持多种量化方案）")
    parser.add_argument(
        "--mode", choices=["fp", "i8", "all"], default="i8",
        help="量化方案: fp=FP16, i8=INT8, all=全部生成 (默认: i8)",
    )
    args = parser.parse_args()

    modes = QUANT_MODES if args.mode == "all" else {args.mode: QUANT_MODES[args.mode]}
    results = {}

    for name, cfg in modes.items():
        results[name] = convert(name, cfg)

    # 汇总
    print(f"\n{'='*60}")
    print("  转换结果汇总")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "成功" if ok else "失败"
        print(f"  {name:6s} | {status} | {QUANT_MODES[name]['output']}")
    print()

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
