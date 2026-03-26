# RK3588 部署指南 — 给板端 Claude 的操作指令

请按顺序执行以下步骤。每完成一步，确认结果正常后再进入下一步。

---

## Step 0: 环境确认

```bash
dpkg --print-architecture     # 应输出 arm64
python3 --version             # 需要 3.8+
cat /sys/class/misc/npu/device/uevent 2>/dev/null || dmesg | grep -i rknpu | head -3
```

---

## Step 1: 拉取代码

```bash
cd ~
git clone https://github.com/Subaru486desuwa/power-line-detection.git
cd power-line-detection
sudo apt install git-lfs -y && git lfs install && git lfs pull

# 验证大文件已下载（不应该是几百字节的指针文件）
ls -lh deploy_rk3588/model/best.onnx
# 应显示约 28MB
```

---

## Step 2: 安装依赖

```bash
sudo apt install -y python3-pip python3-opencv libgl1-mesa-glx

pip3 install numpy opencv-python Pillow
pip3 install rknn-toolkit-lite2
pip3 install PySide6
```

如果 pip 安装超时，使用清华镜像：`pip3 install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`

验证安装：
```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknn OK')"
python3 -c "import cv2; print('opencv OK')"
python3 -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
```

---

## Step 3: 转换模型（ONNX → RKNN）

```bash
cd ~/power-line-detection/deploy_rk3588

# 生成全部版本（INT8 + FP16），用于对比精度差异
python3 convert_rknn.py --mode all

# 或只生成单个：--mode i8 / --mode fp
```

> **注意：** 转换需要 rknn-toolkit2（不是 lite2）。如果板端只有 lite2 无法转换，
> 需要在 x86 PC 上转好 .rknn 文件后 scp 传过来。遇到此情况请告知用户。

确认：`ls -lh model/*.rknn`

---

## Step 4: 命令行推理测试（重点：对比精度差异）

用 6 张精选图片分别测试 i8 和 fp16 的效果差异。

```bash
cd ~/power-line-detection/deploy_rk3588
mkdir -p results

# INT8 版本
for img in ../demo_app/sample_images/*.jpg; do
    name=$(basename "$img" .jpg)
    python3 inference_rk3588.py --img "$img" --model model/best_i8.rknn --save "results/${name}_i8.jpg"
done

# FP16 版本
for img in ../demo_app/sample_images/*.jpg; do
    name=$(basename "$img" .jpg)
    python3 inference_rk3588.py --img "$img" --model model/best_fp16.rknn --save "results/${name}_fp16.jpg"
done
```

**重点关注并记录：**
- 两个版本的检测数量是否一致
- 置信度分数差异
- 是否有漏检或误检
- 推理耗时对比

---

## Step 5: 启动 GUI 演示

```bash
cd ~/power-line-detection/demo_app
python3 main.py
```

- 自动检测到 rknnlite 使用 RKNN 后端
- 状态栏显示 "模型: YOLOv5s (rknn)"
- 点"图片"按钮，选 sample_images/ 下的图测试
- 没有显示器就跳过，只用 Step 4 命令行测试

---

## Step 6: 打包结果推回 Git

```bash
cd ~/power-line-detection
git add deploy_rk3588/results/
git commit -m "feat: RK3588 推理结果图（i8 vs fp16 对比）"
git push
```

---

## 常见问题

**rknn-toolkit-lite2 安装失败：**
`pip3 install rknn-toolkit-lite2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

**PySide6 ARM64 无 wheel：**
`sudo apt install python3-pyside6 -y` 或跳过 GUI 只用命令行

**NPU 推理 init_runtime failed：**
```bash
sudo dmesg | grep -i rknpu
ls /dev/dri/
sudo chmod 666 /dev/dri/renderD128
```

**git lfs pull 失败：**
手动下载 ONNX 到 `deploy_rk3588/model/best.onnx`
