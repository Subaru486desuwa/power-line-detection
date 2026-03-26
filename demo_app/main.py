"""电力线路异物检测系统 — PySide6 演示主窗口。"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QStatusBar, QToolBar, QFileDialog, QSplitter, QFrame,
)

from resources import CLASS_COLORS, CLASS_NAMES_CN
from detector import Detector


# ─────────────────── 工具函数 ───────────────────

def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """BGR ndarray → QPixmap（RGB 转换 + 深拷贝）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ─────────────────── 检测线程 ───────────────────

class DetectionWorker(QThread):
    """在子线程中循环读帧 + 推理，通过信号发回主线程。"""

    frame_ready = Signal(np.ndarray, dict, float)  # annotated, stats, ms
    finished = Signal()

    def __init__(self, detector: Detector, source):
        super().__init__()
        self._detector = detector
        self._source = source  # 视频路径(str) 或摄像头索引(int)
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self.finished.emit()
            return

        while self._running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            annotated, stats, ms = self._detector.detect(frame)
            self.frame_ready.emit(annotated, stats, ms)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False


# ─────────────────── 统计面板 ───────────────────

class StatsPanel(QFrame):
    """右侧检测结果统计面板。"""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background: #1e1e2e; border-radius: 8px;")
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # 标题
        title = QLabel("检测结果")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(title)

        # 类别行
        self._class_labels: dict[str, QLabel] = {}
        for name_en, name_cn in CLASS_NAMES_CN.items():
            row = QHBoxLayout()
            # 色块
            dot = QLabel()
            dot.setFixedSize(14, 14)
            r, g, b = CLASS_COLORS[name_en][::-1]  # BGR→RGB
            dot.setStyleSheet(
                f"background: rgb({r},{g},{b}); border-radius: 7px;"
            )
            row.addWidget(dot)
            # 类名 + 计数
            lbl = QLabel(f" {name_cn}:  0")
            lbl.setFont(QFont("Microsoft YaHei", 15))
            lbl.setStyleSheet("color: #cdd6f4;")
            row.addWidget(lbl)
            row.addStretch()
            layout.addLayout(row)
            self._class_labels[name_en] = lbl

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #45475a;")
        layout.addWidget(sep)

        # 总计 + 耗时
        self._total_label = QLabel("总计: 0")
        self._total_label.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        self._total_label.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(self._total_label)

        self._time_label = QLabel("推理耗时: --")
        self._time_label.setFont(QFont("Microsoft YaHei", 14))
        self._time_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._time_label)

        layout.addStretch()

        # 置信度滑块
        conf_title = QLabel("置信度阈值")
        conf_title.setFont(QFont("Microsoft YaHei", 13))
        conf_title.setStyleSheet("color: #a6adc8;")
        layout.addWidget(conf_title)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 90)
        self.slider.setValue(50)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #45475a; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 16px; margin: -5px 0; border-radius: 8px; }
        """)
        slider_row.addWidget(self.slider)

        self._conf_label = QLabel("0.50")
        self._conf_label.setFont(QFont("Microsoft YaHei", 13, QFont.Weight.Bold))
        self._conf_label.setStyleSheet("color: #89b4fa;")
        self._conf_label.setFixedWidth(40)
        slider_row.addWidget(self._conf_label)
        layout.addLayout(slider_row)

    def update_stats(self, stats: dict, elapsed_ms: float):
        total = 0
        for name_en, count in stats.items():
            cn = CLASS_NAMES_CN.get(name_en, name_en)
            self._class_labels[name_en].setText(f" {cn}:  {count}")
            total += count
        self._total_label.setText(f"总计: {total}")
        self._time_label.setText(f"推理耗时: {elapsed_ms:.1f}ms")

    def update_conf_display(self, value: int):
        self._conf_label.setText(f"{value / 100:.2f}")


# ─────────────────── 主窗口 ───────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("电力线路异物检测系统")
        self.resize(1280, 800)
        self.setStyleSheet("background: #11111b;")

        self._worker: DetectionWorker | None = None
        self._fps_counter: list[float] = []

        # 初始化检测器
        self._detector = Detector(conf=0.50)

        self._init_toolbar()
        self._init_central()
        self._init_statusbar()

    # ---- 界面构建 ----

    def _init_toolbar(self):
        tb = QToolBar("工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        tb.setStyleSheet("""
            QToolBar { background: #1e1e2e; spacing: 8px; padding: 4px 8px; border: none; }
            QToolButton {
                color: #cdd6f4; background: #313244; border-radius: 6px;
                padding: 6px 16px; font-size: 14px; font-family: 'Microsoft YaHei';
            }
            QToolButton:hover { background: #45475a; }
        """)

        self._act_image = tb.addAction("📷 图片")
        self._act_video = tb.addAction("🎬 视频")
        self._act_camera = tb.addAction("📹 摄像头")
        self._act_stop = tb.addAction("⏹ 停止")
        self._act_stop.setEnabled(False)

        self._act_image.triggered.connect(self._on_image)
        self._act_video.triggered.connect(self._on_video)
        self._act_camera.triggered.connect(self._on_camera)
        self._act_stop.triggered.connect(self._on_stop)

        self.addToolBar(tb)

    def _init_central(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #181825; width: 2px; }")

        # 左：图像显示
        self._display = QLabel("请选择图片或视频开始检测")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setFont(QFont("Microsoft YaHei", 18))
        self._display.setStyleSheet("color: #585b70; background: #181825; border-radius: 8px;")
        self._display.setMinimumSize(640, 480)

        # 右：统计面板
        self._stats = StatsPanel()
        self._stats.slider.valueChanged.connect(self._on_conf_changed)

        splitter.addWidget(self._display)
        splitter.addWidget(self._stats)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _init_statusbar(self):
        sb = QStatusBar()
        sb.setStyleSheet(
            "QStatusBar { background: #1e1e2e; color: #a6adc8; font-size: 13px; "
            "font-family: 'Microsoft YaHei'; padding: 2px 8px; }"
        )
        device = self._detector.device_name
        sb.showMessage(f"GPU: {device}  |  模型: YOLOv5s  |  FPS: --")
        self._statusbar = sb
        self.setStatusBar(sb)

    # ---- 事件处理 ----

    @Slot()
    def _on_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(Path.home()),
            "图片 (*.jpg *.jpeg *.png *.bmp)",
        )
        if not path:
            return
        self._stop_worker()
        frame = cv2.imread(path)
        if frame is None:
            return
        annotated, stats, ms = self._detector.detect(frame)
        self._show_frame(annotated)
        self._stats.update_stats(stats, ms)
        self._update_status(ms)

    @Slot()
    def _on_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(Path.home()),
            "视频 (*.mp4 *.avi *.mkv *.mov)",
        )
        if not path:
            return
        self._start_worker(path)

    @Slot()
    def _on_camera(self):
        self._start_worker(0)

    @Slot()
    def _on_stop(self):
        self._stop_worker()

    @Slot(int)
    def _on_conf_changed(self, value: int):
        conf = value / 100
        self._detector.set_conf(conf)
        self._stats.update_conf_display(value)

    # ---- 视频/摄像头线程管理 ----

    def _start_worker(self, source):
        self._stop_worker()
        self._worker = DetectionWorker(self._detector, source)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()
        self._act_stop.setEnabled(True)

    def _stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None
        self._act_stop.setEnabled(False)

    @Slot(np.ndarray, dict, float)
    def _on_frame_ready(self, annotated, stats, ms):
        self._show_frame(annotated)
        self._stats.update_stats(stats, ms)
        self._update_status(ms)

    @Slot()
    def _on_worker_done(self):
        self._act_stop.setEnabled(False)

    # ---- 辅助 ----

    def _show_frame(self, bgr: np.ndarray):
        pix = bgr_to_qpixmap(bgr)
        scaled = pix.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)

    def _update_status(self, ms: float):
        fps = 1000 / ms if ms > 0 else 0
        device = self._detector.device_name
        self._statusbar.showMessage(
            f"GPU: {device}  |  模型: YOLOv5s  |  FPS: {fps:.0f}"
        )

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()


# ─────────────────── 入口 ───────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局深色调色板
    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#11111b"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#cdd6f4"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#cdd6f4"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#313244"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#cdd6f4"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#89b4fa"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
