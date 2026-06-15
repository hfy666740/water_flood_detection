# YOLO 水淹车辆检测系统

基于 YOLO 的智能水淹车辆检测与分析系统，支持目标检测和语义分割功能。

## 🚀 功能特点

- **目标检测**：使用 YOLO 模型检测图像中的车辆，并识别被水淹没的车辆位置
- **语义分割**：对图像进行语义分割，分析水淹区域
- **可视化展示**：蓝色边框标注检测到的车辆，显示淹没位置数量统计
- **置信度分析**：计算并展示平均置信度和最大置信度
- **数据持久化**：检测结果自动保存到 SQLite 数据库
- **结果导出**：支持导出检测结果为文本文件

## 🛠️ 技术栈

- **框架**: Gradio 5.x
- **深度学习**: PyTorch 2.8 + YOLOv8
- **图像处理**: OpenCV, PIL
- **数据库**: SQLite
- **语言**: Python 3.8+

## 📦 依赖安装

### 方法一：使用 pip 安装

```bash
pip install -r requirements.txt
```

### 方法二：使用 Conda 环境（推荐）

```bash
# 创建环境
conda create -n yolo_latest python=3.10

# 激活环境
conda activate yolo_latest

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表

| 库名 | 版本 | 说明 |
|------|------|------|
| gradio | 5.49.1 | Web 界面框架 |
| torch | 2.8.0+cu126 | PyTorch 深度学习框架 |
| torchvision | 0.23.0+cu126 | 计算机视觉工具库 |
| ultralytics | 8.3.200 | YOLO 模型库 |
| opencv_python | 4.12.0.88 | OpenCV 图像处理 |
| Pillow | 12.0.0 | PIL 图像处理 |
| numpy | 2.3.4 | 数值计算 |
| matplotlib | 3.10.7 | 可视化 |
| pycocotools | 2.0.10 | COCO 数据集工具 |
| scipy | 1.16.2 | 科学计算 |
| tqdm | 4.67.1 | 进度条 |
| onnx | 1.19.1 | ONNX 模型支持 |

## 🚗 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/YOLO_Flooding.git
cd YOLO_Flooding
```

### 2. 下载预训练模型

将 YOLO 预训练模型放置在 `detection_models/` 目录下：

- **推荐模型**: `yolov8m.pt` (中等规模模型，平衡速度和精度)
- **小型模型**: `yolov8n.pt` (快速但精度较低)
- **世界模型**: `yolov8s-worldv2.pt` (支持开放词汇检测)

### 3. 运行项目

```bash
# 激活 Conda 环境（如果使用）
conda activate yolo_latest

# 运行主程序
python main.py
```

### 4. 访问界面

运行后在浏览器中打开: http://localhost:7860

## 📁 项目结构

```
YOLO_Flooding/
├── assets/              # 静态资源文件
├── detection_models/    # 目标检测模型目录
├── load_models/         # 传统检测模型（Faster-RCNN, SSD等）
├── load_seg_models/     # 语义分割模型
├── outputs/             # 检测结果输出目录
├── segmentation_models/  # 分割模型目录
├── src/                 # 测试数据
├── tools/               # 工具脚本
├── weights/             # 预训练权重
├── database.py          # 数据库操作模块
├── main.py              # 主程序入口
├── requirements.txt     # 依赖列表
└── YOLO_flooded.db      # SQLite 数据库文件
```

## 📖 使用说明

### 目标检测

1. 在左侧选择"目标检测"功能
2. 上传包含车辆的图像
3. 选择检测模型（推荐使用 yolo(8m).pt）
4. 点击"开始检测"按钮
5. 查看检测结果：
   - 蓝色边框标注的车辆
   - 右下角显示淹没车辆数量统计
   - 平均置信度和最大置信度

### 语义分割

1. 在左侧选择"语义分割"功能
2. 上传图像
3. 选择分割模型
4. 查看分割结果

### 结果保存

- 检测结果自动保存到 `YOLO_flooded.db` 数据库
- 分析结果文件保存到 `outputs/` 目录
- 支持下载检测结果为 TXT 文件

## 📊 检测结果说明

| 统计项 | 说明 |
|--------|------|
| 淹没车辆数量 | 检测到的被水淹没的车辆总数 |
| 平均置信度 | 所有检测框的平均置信度 |
| 最大置信度 | 检测框中的最高置信度 |

## 🛡️ 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*项目基于 YOLOv8 构建，感谢 Ultralytics 团队的优秀工作*
