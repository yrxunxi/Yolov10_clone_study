# YOLOv10 道路车辆目标检测项目说明书

## 1. 引言

本说明书旨在为 YOLOv10 道路车辆目标检测项目提供详尽的指导，帮助使用者理解项目代码结构、配置运行环境、执行模型训练与评估，并进行目标检测推理。本项目基于最新的 **YOLOv10n** 模型，在 **Road Vehicle Images Dataset** 上进行了训练，以实现对道路场景中各类车辆的准确识别与定位。本说明书将分别提供在**本地环境**和 **Google Colab 挂载 Google Drive 环境**下的操作指南。

------

## 2. 环境配置

本项目主要在 **Google Colab Pro** 环境下使用 **NVIDIA Tesla T4 GPU** 进行训练。但代码同样支持在配置了合适 GPU 的本地环境运行。为确保代码顺利运行，请根据您的环境选择相应部分进行配置。

### 2.1 硬件要求

- **GPU：** 建议使用 NVIDIA GPU，至少 8GB 显存，推荐 NVIDIA Tesla T4 (Colab) 或 NVIDIA RTX 30系列/40系列 (本地) 或更高。
- **内存：** 至少 12GB RAM。
- **磁盘空间：** 至少 10GB 可用空间用于存储数据集、模型权重和结果。

### 2.2 软件要求

- **操作系统：** Windows 10/11, Ubuntu 20.04+ 或 macOS。
- **Python 版本：** Python 3.9+ (建议 3.9 或 3.10)。
- **CUDA Toolkit：** 适配您的 NVIDIA GPU 驱动的 CUDA 版本（例如，CUDA 11.8 或 CUDA 12.1）。在 Colab 环境下，CUDA 通常已预配置。
- **PyTorch：** 兼容的 PyTorch 版本。

### 2.3 依赖库安装

推荐使用 `pip` 或 `conda` 进行依赖库的安装。

#### 2.3.1 本地环境安装步骤

1. **克隆项目仓库 (如果您的代码在 GitHub 或其他仓库中)：**

   Bash

   ```
   git clone [你的GitHub仓库链接]
   cd [你的项目根目录，例如：yolov10-traffic-detection]
   ```

   **【如果你的代码是直接压缩包提供，请说明解压到本地某个目录】**

2. **创建并激活虚拟环境 (推荐)：**

   Bash

   ```
   # 使用 venv
   python -m venv venv
   source venv/bin/activate # Linux/macOS
   # 或 venv\Scripts\activate # Windows
   
   # 或使用 conda
   conda create -n yolov10_env python=3.9
   conda activate yolov10_env
   ```

3. **安装 PyTorch (根据你的 CUDA 版本选择对应命令)：**

   - CUDA 11.8：

     Bash

     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

   - CUDA 12.1+：

     Bash

     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

   - 仅 CPU 版本 (不推荐用于训练)：

     Bash

     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

4. **安装 Ultralytics (YOLOv10 核心库) 及其他依赖：**

   Bash

   ```
   pip install ultralytics opencv-python matplotlib seaborn Pillow
   ```

   您也可以在本地项目根目录中生成一个 `requirements.txt` 文件来一次性安装所有依赖：

   Bash

   ```
   pip freeze > requirements.txt
   # 然后在目标环境安装：
   pip install -r requirements.txt
   ```

#### 2.3.2 Google Colab 环境安装步骤

在 Colab Notebook 中运行以下代码块：

1. **挂载 Google Drive 并导航到项目目录：**

   Python

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   
   # 导航到你的项目目录 (根据实际路径修改)
   # 例如，如果你的项目在 Google Drive 的 MyDrive/YOLOv10_Project
   %cd /content/drive/MyDrive/YOLOv10_Project/
   ```

   **【请务必根据你的实际 Google Drive 项目路径修改 `%cd` 命令】**

2. **安装 Ultralytics 及其他依赖：**

   Bash

   ```
   !pip install ultralytics opencv-python matplotlib seaborn Pillow
   ```

   **注意：** Colab 环境通常已预装 PyTorch 和 CUDA，因此通常不需要单独安装 PyTorch。

------

## 3. 项目结构

项目的典型文件和文件夹结构如下：

```
your_project_root/
├── colab_trafic_data_config.yaml  # 数据集配置文件
├── yolov10n.pt                    # YOLOv10n 预训练权重文件 (如果直接下载到本地，Colab运行时会自动下载)
├── runs/                          # 训练和验证结果输出目录
│   └── detect/
│       └── yolov10n_colab_run1/   # 你的训练实验名称
│           ├── weights/           # 训练好的模型权重 (last.pt, best.pt)
│           ├── results.txt        # 训练日志
│           ├── results.png        # 训练曲线图
│           ├── F1_curve.png       # F1 曲线图
│           ├── P_curve.png        # 精确率曲线图
│           ├── R_curve.png        # 召回率曲线图
│           ├── PR_curve.png       # P-R 曲线图
│           ├── confusion_matrix.png     # 混淆矩阵
│           ├── confusion_matrix_normalized.png # 归一化混淆矩阵
│           ├── train_batch*.jpg   # 训练批次示例图
│           └── val_batch*.jpg     # 验证批次预测/标签示例图
└── [你的Colab Notebook文件.ipynb] # (如果使用Jupyter Notebook)
└── [你的Python脚本文件.py]       # (如果使用Python脚本)
```

**【注意：在 Colab 挂载 Google Drive 环境下，你的数据集 `images/` 和 `labels/` 文件夹通常直接存在于 Google Drive 上的某个路径，而不是复制到项目根目录下。`colab_trafic_data_config.yaml` 文件中的 `path` 字段将指向 Google Drive 上的数据集根目录。】**

## 4. 数据集准备

本项目使用 Kaggle 上的 **Road Vehicle Images Dataset**。

1. 下载数据集：

   访问 Kaggle 网站：https://www.kaggle.com/datasets/hasibullahh/road-vehicle-images-dataset

   您需要注册 Kaggle 账号并同意数据集使用条款才能下载。

2. 组织数据集：

   下载并解压数据集后，您会得到 train 和 val 两个文件夹。每个文件夹内包含 images 和 labels 子文件夹。

   - 对于本地环境：

     请将解压后的数据集根目录（包含 train 和 val 文件夹的上一级目录）放置在您的项目目录下的 data/ 文件夹中（例如：your_project_root/data/RoadVehicleImagesDataset/），或您希望的任何本地路径。

   - 对于 Google Colab 挂载 Google Drive 环境：

     请将解压后的数据集根目录直接上传到您的 Google Drive 中的某个位置（例如：MyDrive/datasets/road_vehicle_images_dataset_root/）。

3. 配置数据集 YAML 文件 (colab_trafic_data_config.yaml)：

   此文件告知 YOLO 模型数据集的路径和类别信息。

   **`colab_trafic_data_config.yaml` 示例内容 (请根据您的实际路径和类别修改)：**

   YAML

   ```
   # dataset root dir
   # 本地环境示例 (如果数据集在 your_project_root/data/RoadVehicleImagesDataset)
   # path: ./data/RoadVehicleImagesDataset
   
   # Google Colab 挂载 Google Drive 环境示例 (如果数据集在 MyDrive/datasets/road_vehicle_images_dataset_root)
   path: /content/drive/MyDrive/datasets/road_vehicle_images_dataset_root
   
   train: images/train  # 训练图像路径 (相对于 'path')
   val: images/val      # 验证图像路径 (相对于 'path')
   test:                # 如果有测试集，可在此处指定
   
   # Classes
   names:
     0: car
     1: truck
     2: bus
     3: motorbike
     # ... 【请在此处列出所有实际的类别名称和 ID，从0开始】
   ```

   **【请务必根据你的实际数据集存储位置和类别信息修改 `colab_trafic_data_config.yaml` 中的 `path` 和 `names` 字段。】**

## 5. 运行程序

本项目的主要操作通过 Ultralytics 的 `yolo` 命令行接口完成。

### 5.1 模型训练

使用以下命令开始模型训练。

- **对于本地环境 (在终端中运行)：**

  Bash

  ```
  yolo detect train data=./colab_trafic_data_config.yaml model=yolov10n.pt epochs=50 batch=32 imgsz=640 device=0 amp=True workers=4 name=yolov10n_local_run1
  ```

  **注意：** 如果你的 `colab_trafic_data_config.yaml` 中的 `path` 指向的是本地路径，确保它是正确的相对或绝对路径。

- 对于 Google Colab 挂载 Google Drive 环境 (在 Colab Notebook 中运行)：

  在运行此命令前，确保你已经成功挂载 Google Drive (from google.colab import drive; drive.mount(...))，并且已经使用 %cd 命令导航到项目根目录。

  Bash

  ```
  !yolo detect train data=./colab_trafic_data_config.yaml model=yolov10n.pt epochs=50 batch=32 imgsz=640 device=0 amp=True workers=4 name=yolov10n_colab_run1
  ```

  **注意：** 确保你的 `colab_trafic_data_config.yaml` 中的 `path` 指向的是 Google Drive 上的数据集路径（例如 `/content/drive/MyDrive/datasets/...`）。

**参数说明：**

- `data=[path/to/config.yaml]`: 指定数据集配置文件的路径。
- `model=yolov10n.pt`: 指定预训练模型权重。`yolov10n.pt` 是 YOLOv10n 在 COCO 数据集上预训练的权重。如果本地或 Colab 环境中没有，Ultralytics 库会自动下载。
- `epochs=50`: 设定训练的总轮次为 50。
- `batch=32`: 设定每个训练批次的大小为 32。
- `imgsz=640`: 设定输入图像的尺寸为 640x640 像素。
- `device=0`: 指定训练设备为第一个 GPU (编号为 0)。如果您有多个 GPU，可以更改此值。如果您的系统没有 GPU，请使用 `device=cpu` (但训练速度会非常慢)。
- `amp=True`: 启用自动混合精度训练，可以加速训练并减少显存消耗。
- `workers=4`: 指定数据加载的子进程数为 4，提高数据加载效率。
- `name=[experiment_name]`: 指定本次实验的名称。所有结果将保存在 `runs/detect/[experiment_name]/` 目录下。

训练过程中，会在终端实时显示损失、mAP 等指标的变化。训练结束后，最终的模型权重 (如 `best.pt` 和 `last.pt`) 将保存在 `runs/detect/[experiment_name]/weights/` 目录下。

### 5.2 模型验证 (评估)

训练完成后，可以使用训练好的模型在验证集上进行评估，以获取详细的性能指标。

- **对于本地环境：**

  Bash

  ```
  yolo detect val model=runs/detect/yolov10n_local_run1/weights/best.pt data=./colab_trafic_data_config.yaml
  ```

- **对于 Google Colab 挂载 Google Drive 环境：**

  Bash

  ```
  !yolo detect val model=runs/detect/yolov10n_colab_run1/weights/best.pt data=./colab_trafic_data_config.yaml
  ```

**参数说明：**

- `model=[path/to/weights.pt]`: 指定用于验证的模型权重路径。推荐使用训练过程中性能最好的 `best.pt`。
- `data=[path/to/config.yaml]`: 再次指定数据集配置，确保验证使用正确的数据集。

验证结果（包括 mAP、精确率、召回率、混淆矩阵等）将显示在终端，并保存在实验目录中。

### 5.3 目标检测推理 (预测)

使用训练好的模型对新的图像或视频进行目标检测。

- **对于本地环境：**

  Bash

  ```
  # 预测单张图片
  yolo detect predict model=runs/detect/yolov10n_local_run1/weights/best.pt source=./test_images/my_car_pic.jpg --conf 0.25 --iou 0.7
  
  # 预测一个文件夹中的所有图片
  yolo detect predict model=runs/detect/yolov10n_local_run1/weights/best.pt source=./test_images/ --conf 0.25 --iou 0.7
  
  # 预测视频文件
  yolo detect predict model=runs/detect/yolov10n_local_run1/weights/best.pt source=./test_videos/my_traffic_clip.mp4 --conf 0.25
  ```

- **对于 Google Colab 挂载 Google Drive 环境：**

  Bash

  ```
  # 预测单张图片
  !yolo detect predict model=runs/detect/yolov10n_colab_run1/weights/best.pt source=/content/drive/MyDrive/test_images/my_car_pic.jpg --conf 0.25 --iou 0.7
  
  # 预测一个文件夹中的所有图片
  !yolo detect predict model=runs/detect/yolov10n_colab_run1/weights/best.pt source=/content/drive/MyDrive/test_images/ --conf 0.25 --iou 0.7
  
  # 预测视频文件
  !yolo detect predict model=runs/detect/yolov10n_colab_run1/weights/best.pt source=/content/drive/MyDrive/test_videos/my_traffic_clip.mp4 --conf 0.25
  ```

**参数说明：**

- `model=[path/to/weights.pt]`: 指定用于推理的模型权重。
- `source=[input_path]`: 指定输入源，可以是图片路径、文件夹路径、视频路径，甚至是摄像头 (`source=0`)。
- `--conf 0.25`: 设置检测结果的置信度阈值。只有置信度高于此值的目标才会被显示。
- `--iou 0.7`: 设置非极大值抑制 (NMS) 的 IoU 阈值。用于过滤重叠的边界框。

预测结果（带有边界框和标签的图像/视频）将保存在 `runs/detect/predict/` 或类似名称的目录下。

## 6. 结果分析

训练和验证过程中生成的各种图表和日志文件是分析模型性能的关键。它们都存储在 `runs/detect/[experiment_name]/` 目录下。

- `results.txt`: 包含每个 Epoch 的详细训练和验证指标（损失、mAP、精确率、召回率等）。
- `results.png`: 汇总了训练过程中所有关键指标随 Epoch 变化的曲线图。
- `P_curve.png`, `R_curve.png`, `F1_curve.png`, `PR_curve.png`: 精确率、召回率、F1-score 和 P-R 曲线图，用于深入分析模型性能。
- `confusion_matrix.png` 和 `confusion_matrix_normalized.png`: 混淆矩阵，直观展示模型在不同类别上的分类表现和混淆情况。
- `labels.jpg`, `labels_correlogram.jpg`: 数据集的标注统计和相关性图，用于理解数据分布。
- `train_batch*.jpg`: 训练批次示例图，展示数据增强效果。
- `val_batch*_labels.jpg` 和 `val_batch*_pred.jpg`: 验证批次中真实标签和模型预测结果的对比图，用于定性分析。

这些图表是撰写项目报告（论文）时的重要依据。

## 7. 常见问题与故障排除

- CUDA 内存不足 (CUDA out of memory)：
  - 尝试减小 `batch` 参数的值（例如，从 32 减小到 16 或 8）。
  - 确保没有其他程序占用 GPU 显存。
  - 在 Colab 中，确保选择了带有足够显存的 GPU 类型（如 T4）。
- 环境配置错误：
  - 检查 Python 版本是否正确。
  - 确保所有依赖库都已正确安装，特别是 PyTorch 和 Ultralytics。
  - 检查 CUDA 版本是否与 PyTorch 版本兼容。
- 文件/路径未找到：
  - 仔细检查 `data=`、`model=` 和 `source=` 参数中的文件或文件夹路径是否正确。
  - 在 Colab 中，确保 Google Drive 已正确挂载，且路径是绝对路径或相对于 Colab Notebook 运行目录的正确相对路径。
  - `colab_trafic_data_config.yaml` 中的 `path` 也需要根据实际数据集位置进行调整。
- 训练性能不佳 (mAP 低)：
  - 增加 `epochs` 数量，进行更长时间的训练。
  - 检查数据集质量和标注是否准确。
  - 调整学习率 (`lr0`) 或优化器。
  - 尝试更复杂的数据增强策略。
  - 考虑使用更大规模的 YOLOv10 模型 (如 YOLOv10m 或 YOLOv10l)，如果计算资源允许。

------

**提交要求：**

- **电子版：** 请将本说明书以 Markdown (`.md`) 或 PDF 格式提交。
- **纸质版：** 请打印 PDF 版本的说明书。

------

希望这份详细的程序说明书能完全满足你的需求！请务必根据你的**实际项目路径、数据集详情（类别名称、数量）\**和你在 Colab Notebook 中的\**具体操作**（例如 Google Drive 挂载方式，以及你的代码是直接放在 Colab Notebook 中还是从 Drive 某个文件夹运行）来**填充和调整**相应的占位符和描述。
