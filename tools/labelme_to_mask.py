import os
import json
from PIL import Image, ImageDraw
import numpy as np

# 类别名称，从0开始，0为背景（mask图中为1开始）
CLASS_NAMES = ['water']

# 路径配置
IMAGE_ROOT = "path/to/your/dataset/images"  # 修改为你的图像数据集路径  
LABEL_ROOT = "path/to/your/dataset/labels"  # 修改为你的LabelMe JSON标签路径
SAVE_ROOT = "path/to/your/dataset/masks"# 修改为你的掩码图像保存路径
SPLITS = ['train', 'val', 'test']

def make_mask_from_json(image_path: str, json_path: str, save_path: str):
    image = Image.open(image_path)
    width, height = image.size
    mask = Image.new('P', (width, height))  # 单通道P模式

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in CLASS_NAMES:
            continue  
        points = shape['points']
        polygon = tuple(tuple(p) for p in points)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=CLASS_NAMES.index(label) + 1)  # 从1开始编码

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask.save(save_path)


def process_dataset():
    for split in SPLITS:
        image_dir = os.path.join(IMAGE_ROOT, split)
        label_dir = os.path.join(LABEL_ROOT, split)
        save_dir = os.path.join(SAVE_ROOT, split)

        image_list = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

        for img_name in image_list:
            base_name = os.path.splitext(img_name)[0]
            image_path = os.path.join(image_dir, img_name)
            json_path = os.path.join(label_dir, base_name + '.json')
            save_path = os.path.join(save_dir, base_name + '.png')

            if not os.path.exists(json_path):
                print(f"Warning: JSON label not found for {img_name}, skipping...")
                continue

            make_mask_from_json(image_path, json_path, save_path)
            print(f"Saved mask: {save_path}")


def vis_label(mask_path: str):
    img = Image.open(mask_path)
    img_np = np.array(img)
    unique_values = set(img_np.flatten().tolist())
    print(f"{mask_path} 包含的类别索引: {unique_values}")


if __name__ == '__main__':
    process_dataset()
