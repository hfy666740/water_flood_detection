import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple
try:
    from .Net import SegNet  # 如果 SegNet.py 在 load_models 文件夹内
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from Net import SegNet

def load_model(pt_path: str, num_classes: int = 2):
    model = SegNet(num_classes=num_classes)
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(image: Image.Image, model, device: str = '0'):
    """
    执行分割预测并返回结果
    返回格式：(processed_image, result_data)
    result_data 是包含统计信息的字典
    """
    # 设置设备
    if device == '0':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    model.to(device_obj)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device_obj)
    
    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    # 生成高亮可视化图像
    result_image = create_water_highlight_image(image, pred_mask, original_size)
    
    # 计算像素比例
    mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
    mask_array = np.array(mask_resized)
    
    total_pixels = mask_array.size
    water_pixels = np.sum(mask_array == 1)
    bg_pixels = total_pixels - water_pixels
    
    background_ratio = f"{(bg_pixels / total_pixels):.2%}"
    water_ratio = f"{(water_pixels / total_pixels):.2%}"
    
    # 返回结果数据字典（与 YOLO 分割格式一致）
    result_data = {
        "background_ratio": background_ratio,
        "water_ratio": water_ratio
    }
    
    return result_image, result_data

def create_water_highlight_image(original: Image.Image, mask: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
    """创建水面高亮图像"""
    mask_resized = Image.fromarray(mask.astype(np.uint8)).resize(target_size, Image.NEAREST)
    mask_array = np.array(mask_resized)
    original_array = np.array(original.convert('RGB'))
    
    result_array = original_array.copy()
    water_mask = mask_array == 1
    red_overlay = np.array([0, 128, 0])  # 暗红色
    alpha = 0.6
    
    for c in range(3):
        result_array[water_mask, c] = (
            (1 - alpha) * original_array[water_mask, c] +
            alpha * red_overlay[c]
        ).astype(np.uint8)
    
    return Image.fromarray(result_array)