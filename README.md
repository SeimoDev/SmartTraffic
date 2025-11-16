# Yolo 车辆计数（中文说明）

## 概览
- 本项目使用 YOLOv3 进行实时交通视频中的车辆检测与计数，支持的类别包括：`bicycle`、`car`、`motorbike`、`bus`、`truck`、`train`。
- 检测后通过最近帧匹配保持 ID 一致，避免同一辆车重复计数。
- 提供命令行与图形界面两种使用方式；支持性能优化选项（输入尺寸、跳帧、是否显示窗口、视频编码等）。

## 核心流程
- 读取视频帧 → 构造 YOLO 输入 `blob` → 前向推理 → 取最大类别概率与阈值过滤 → 坐标反缩放 → 非极大值抑制（NMS） → 绘制框与文本 → 计数与 ID 管理 → 写出视频帧。
- 关键实现位置：
  - 检测前向与解析：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:243-251`
  - NMS 与框绘制：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:303-317`
  - 车辆计数逻辑：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:149-186`
  - 历史帧匹配（KDTree 最近邻）：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:103-123`
  - 异步检测与结果复用（避免卡顿）：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:247-276`

## 环境与依赖
- Python 3.x（Windows 默认自带 `tkinter`）
- 依赖库：
  - `opencv-python`（建议 4.x 及以上）
  - `numpy`
  - `scipy`
- 安装示例：
  ```bash
  pip install opencv-python numpy scipy
  ```

## 模型与数据
- 目录 `yolo-coco` 包含：
  - `coco.names`（COCO 类别名）
  - `yolov3.cfg`（模型结构配置，默认 `width=288, height=288`）
- 权重文件 `yolov3.weights`：
  - 程序会在首次运行时自动下载到 `yolo-coco/yolov3.weights`；离线环境可手动下载。
  - 手动下载示例（PowerShell）：
    ```powershell
    Invoke-WebRequest https://pjreddie.com/media/files/yolov3.weights -OutFile yolo-coco\yolov3.weights
    ```

## 命令行参数说明
- 所有参数由 `input_retrieval.py` 解析：`d:/DeepLearning/Yolo-Vehicle-Counter/input_retrieval.py:18-41`
- 参数列表：
  - `--input` 输入视频路径，必须设置。
  - `--output` 输出视频路径，必须设置。若为 `.mp4`，编码自动使用 `mp4v`。
  - `--yolo` YOLO 目录（含 `coco.names`、`yolov3.cfg`、`yolov3.weights`）。
  - `--confidence` 置信度阈值，过滤低置信度目标，范围 `0~1`（默认 `0.5`）。
  - `--threshold` NMS 阈值，抑制重叠框（默认 `0.3`）。
  - `--use-gpu` 是否启用 GPU 加速，支持 `1/0、true/false…`（默认 `false`）。
    - 若安装的是 CPU 版 OpenCV（官方轮子），会自动回退到 CPU，避免崩溃。
  - `--input-size` YOLO 输入尺寸（正方形，建议用 32 的倍数），默认 `416`。
    - 运行时会自动对齐到 `yolov3.cfg` 的 `width/height`，避免网络层拼接错误。
  - `--skip-frames` 跳帧检测（整数），例如 `1` 表示隔 1 帧检测一次，其余帧复用上次结果；默认 `0`。
  - `--display` 是否显示窗口，`1/0、true/false…`；关闭可减少 GUI 开销；默认 `true`。
  - `--fourcc` 输出编码，默认 `MJPG`；`.mp4` 输出自动改为 `mp4v` 以提升兼容性与速度。

## 使用示例（命令行）
- 基本运行：
  ```powershell
  python yolo_video.py --input .\bridge.mp4 --output outputVideos\out.avi --yolo yolo-coco --confidence 0.3
  ```
- 提升速度（关闭窗口、缩小输入、跳帧）：
  ```powershell
  python yolo_video.py --input .\bridge.mp4 --output outputVideos\fast.avi --yolo yolo-coco --input-size 320 --skip-frames 2 --display 0 --fourcc MJPG
  ```
- 输出 MP4（自动使用 `mp4v` 编码）：
  ```powershell
  python yolo_video.py --input .\bridge.mp4 --output outputVideos\out.mp4 --yolo yolo-coco
  ```

## 图形界面（GUI）
- 运行：
  ```powershell
  python gui.py
  ```
- 界面提供所有参数的可视化设置：选择输入/输出、YOLO 目录、`confidence`、`threshold`、`use-gpu`、`input-size`、`skip-frames`、`display`、`fourcc`，并可实时查看日志与状态。

## 算法与参数取舍
- `confidence` 越高：误检更少，但可能漏检；一般 `0.3~0.6` 之间选择。
- `threshold` 越高：抑制重叠框更强；一般 `0.3~0.5`。
- `input-size` 越小：推理更快，定位更粗糙；常见取值 `288/320/416/608`。
- `skip-frames` 越大：总体吞吐更高；检测帧之间复用最近结果，计数在检测帧更新。
- `display=0`：关闭窗口可显著减少卡顿，专注于推理与写出。
- `fourcc`：在 Windows 上 `MJPG/XVID` 写出 `.avi` 通常更快；`.mp4` 统一用 `mp4v`。

## GPU 加速说明
- 仅在安装了 **CUDA 版本** 的 OpenCV 时，`--use-gpu 1` 才会启用 `DNN_BACKEND_CUDA`。
- 若为官方 CPU 轮子（`opencv-python`），会自动打印警告并回退到 CPU，程序继续运行。
- 相关逻辑：`d:/DeepLearning/Yolo-Vehicle-Counter/yolo_video.py:192-198`

## 常见问题
- 形状不一致（ConcatLayer 错误）：当 `input-size` 与 `yolov3.cfg` 的 `width/height` 不一致或非 32 的倍数，会在前向报错。程序已自动对齐配置尺寸（`yolo_video.py:19-38`）。
- `.mp4` 编码提醒：`MJPG` 与 MP4 容器不匹配时，OpenCV 会降级为 `mp4v`；已自动选择 `mp4v`。
- 权重缺失：首次运行自动下载；离线环境手动放置至 `yolo-coco/yolov3.weights`。
- GPU 报错：未安装 CUDA 版 OpenCV 时启用 GPU 会崩溃；已自动检测并安全回退到 CPU。

## 贡献与许可
- 在使用中若遇到新问题或希望增加新参数，可提交 Issue 或 PR。
- 本项目基于公开数据与预训练模型，合理使用即可。