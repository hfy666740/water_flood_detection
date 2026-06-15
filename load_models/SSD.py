from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import os
import sys

# 将 load_models 目录添加到 Python 路径
LOAD_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if LOAD_MODELS_DIR not in sys.path:
    sys.path.insert(0, LOAD_MODELS_DIR)

from ssd_model import SSD

SSD_LABEL_NAMES = ["bottom", "middle", "top"]

def load_ssd(model_path: str) -> SSD:
    """加载 SSD 模型"""
    base_dir = os.path.dirname(model_path)
    classes_path = os.path.join(LOAD_MODELS_DIR, 'model_data', 'voc_classes.txt')
    
    # 如果 classes 文件不存在，尝试其他路径
    if not os.path.exists(classes_path):
        classes_path = os.path.join(base_dir, 'voc_classes.txt')
    
    # 初始化模型，指定自定义权重路径
    model = SSD(
        model_path=model_path,
        classes_path=classes_path,
        confidence=0.25,
        nms_iou=0.45,
        backbone="vgg",
        cuda=True
    )
    return model


def predict_ssd(image: Image.Image, model: SSD, conf: float = 0.25, iou: float = 0.45) -> Tuple[Image.Image, Dict]:
    """
    使用 SSD 执行推理并返回统一格式的结果
    只推理一次，同时获取可视化图像和统计数据
    
    Args:
        image: 输入图像
        model: SSD模型
        conf: 置信度阈值
        iou: NMS的IoU阈值
    """
    # 导入必要的工具
    from utils2.utils import cvtColor, preprocess_input, resize_image
    from PIL import ImageDraw, ImageFont
    import torch
    
    # 计算输入图片的高和宽
    image_shape = np.array(np.shape(image)[0:2])
    
    # 图像预处理
    image_rgb = cvtColor(image)
    image_data = resize_image(image_rgb, (model.input_shape[1], model.input_shape[0]), model.letterbox_image)
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0
    )
    
    # 执行推理（使用传入的conf和iou参数）
    with torch.no_grad():
        images = torch.from_numpy(image_data).type(torch.FloatTensor)
        if model.cuda:
            images = images.cuda()
        
        outputs = model.net(images)
        results = model.bbox_util.decode_box(
            outputs, model.anchors, image_shape, model.input_shape, 
            model.letterbox_image, nms_iou=iou, confidence=conf  # 使用传入的参数
        )
    
    # 提取统计数据
    b_cnt = m_cnt = t_cnt = 0
    avg_conf = "-"
    max_conf = "-"
    
    if len(results[0]) > 0:
        top_label = np.array(results[0][:, 4], dtype='int32')
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]
        
        for i in range(3):
            count = np.sum(top_label == i)
            if i == 0:
                b_cnt = int(count)
            elif i == 1:
                m_cnt = int(count)
            elif i == 2:
                t_cnt = int(count)
        
        if len(top_conf) > 0:
            avg_conf = f"{np.mean(top_conf):.2%}"
            max_conf = f"{np.max(top_conf):.2%}"
    else:
        top_label = np.array([])
        top_conf = np.array([])
        top_boxes = np.array([])
    
    # 绘制检测框（使用同样的推理结果）
    # 复制原始图像用于绘制
    annotated_image = image.copy()
    
    if len(results[0]) > 0:
        # 设置字体与边框厚度
        font = ImageFont.truetype(
            font=os.path.join(LOAD_MODELS_DIR, 'model_data/simhei.ttf'),
            size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')
        )
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // model.input_shape[0], 1)
        
        # 绘制每个检测框
        for i, c in enumerate(top_label):
            predicted_class = model.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(annotated_image)
            
            # 使用 textbbox 替代已废弃的 textsize
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_size = (label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1])
            label_bytes = label.encode('utf-8')
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            
            # 绘制边框和标签
            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=model.colors[c]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=model.colors[c]
            )
            draw.text(text_origin, str(label_bytes, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
    
    # 构建统计字典
    stats_dict = {
        "b_cnt": b_cnt,
        "m_cnt": m_cnt,
        "t_cnt": t_cnt,
        "avg_conf": avg_conf,
        "max_conf": max_conf
    }
    
    return annotated_image, stats_dict


def load_model(pt_path: str, yaml_path: Optional[str] = None):
    """统一的模型加载接口"""
    return load_ssd(pt_path)


# 修改统一接口
def predict(image: Image.Image, model, conf: float = 0.25, iou: float = 0.45):
    """统一的预测接口"""
    return predict_ssd(image, model, conf=conf, iou=iou)