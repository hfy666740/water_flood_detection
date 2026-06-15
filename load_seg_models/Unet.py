import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple

try:
    from .unet_model import UNet
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from load_seg_models.unet_model import UNet

try:
    from .utils import keep_image_size_open_rgb
    USE_CUSTOM_PREPROCESS = True
except:
    USE_CUSTOM_PREPROCESS = False
from PIL import ImageOps

def pad_to_multiple(image: Image.Image, multiple: int = 32, fill=(0, 0, 0)):
    w, h = image.size
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple
    if pad_w == 0 and pad_h == 0:
        return image, (0, 0)
    # 只在右/下方向补齐，方便后处理直接裁掉
    padded = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=fill)
    return padded, (pad_w, pad_h)

def load_model(pt_path: str, num_classes: int = 2):
    """
    加载 UNet 模型
    
    Args:
        pt_path: 模型权重文件路径
        num_classes: 类别数量（默认2：背景和前景）
    
    Returns:
        加载好的模型（eval模式）
    """
    model = UNet(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print(f"✅ UNet 模型加载成功: {os.path.basename(pt_path)}")
    except Exception as e:
        print(f"❌ UNet 模型加载失败: {e}")
        raise
    
    model.eval()
    return model

def predict(image: Image.Image, model, device: str = '0'):
    """
    执行 UNet 分割预测并返回结果
    
    Args:
        image: PIL Image 对象
        model: 加载好的 UNet 模型
        device: 设备选择（'0' 表示自动选择 cuda:0 或 cpu）
    
    Returns:
        (processed_image, result_data)
        - processed_image: PIL Image，红色叠加的可视化结果
        - result_data: dict，包含 background_ratio 和 water_ratio
    """

    if device == '0':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    model.to(device_obj)

    original_size = image.size  # (width, height)

    # 统一对齐到 32 的倍数，避免 U-Net 拼接尺寸不匹配
    img_aligned, (pad_w, pad_h) = pad_to_multiple(image, multiple=32)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(img_aligned).unsqueeze(0).to(device_obj)
    
    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 裁掉右/下方向的填充，恢复到原图尺寸
    w, h = original_size
    if pad_w or pad_h:
        mask_array = pred_mask[:h, :w]
    else:
        mask_array = pred_mask

    result_image = create_red_overlay_image(image, mask_array)

    total_pixels = mask_array.size
    unique, counts = np.unique(mask_array, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    
    bg_pixels = pixel_counts.get(0, 0)
    fg_pixels = pixel_counts.get(1, 0)
    
    background_ratio = f"{(bg_pixels / total_pixels * 100):.2f}%"
    water_ratio = f"{(fg_pixels / total_pixels * 100):.2f}%"
    
    result_data = {
        "background_ratio": background_ratio,
        "water_ratio": water_ratio
    }
    
    return result_image, result_data

def create_red_overlay_image(original: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    创建红色叠加可视化图像（前景区域显示为红色半透明叠加）
    
    Args:
        original: 原始 PIL Image
        mask: numpy array，值为 0（背景）或 1（前景）
    
    Returns:
        叠加后的 PIL Image
    """
    original_array = np.array(original.convert('RGB'))
    result_array = original_array.copy()
    foreground_mask = (mask == 1)
    red_overlay = np.array([0, 255, 0]) 
    alpha = 0.6  
    for c in range(3):
        result_array[foreground_mask, c] = (
            (1 - alpha) * original_array[foreground_mask, c] +
            alpha * red_overlay[c]
        ).astype(np.uint8)
    
    return Image.fromarray(result_array)