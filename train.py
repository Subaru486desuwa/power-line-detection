"""YOLOv5s 输电线异物检测 — 正式训练脚本。

Usage:
    conda activate dl
    python train.py              # 正式训练 100 epoch
    python train.py --epochs 3   # 快速验证
"""

import subprocess
import sys
from pathlib import Path

# --- 配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
DATA_YAML = PROJECT_ROOT / "data" / "power_line.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

TRAIN_ARGS = {
    "img": 640,
    "batch": 8,         # RTX 3060 Ti 8GB 安全值 (16 会 OOM)
    "epochs": 100,
    "patience": 50,      # EarlyStopping
    "data": str(DATA_YAML),
    "weights": "yolov5s.pt",
    "project": str(RUNS_DIR),
    "name": "power_line",
    "workers": 4,
    "exist-ok": None,    # flag, 无值
}


def build_cmd(overrides: dict | None = None) -> list[str]:
    args = {**TRAIN_ARGS}
    if overrides:
        args.update(overrides)

    cmd = [sys.executable, str(YOLOV5_DIR / "train.py")]
    for k, v in args.items():
        cmd.append(f"--{k}")
        if v is not None:
            cmd.append(str(v))
    return cmd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", type=str, default="power_line")
    cli_args = parser.parse_args()

    overrides = {
        "epochs": cli_args.epochs,
        "batch": cli_args.batch,
        "name": cli_args.name,
    }

    cmd = build_cmd(overrides)
    print(f"[CMD] {' '.join(cmd)}\n")

    # 实时输出到终端，不缓冲
    proc = subprocess.run(cmd, cwd=YOLOV5_DIR)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
