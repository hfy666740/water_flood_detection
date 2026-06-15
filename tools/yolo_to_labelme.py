import os
import glob
import numpy as np
import cv2
import json

def convert_txt_to_labelme_json(txt_dir, img_dir, output_dir, class_names, image_fmt='.jpg'):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

    for txt_file in txt_files:
        txt_name = os.path.basename(txt_file)
        img_name = txt_name.replace(".txt", image_fmt)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[!] 图像文件不存在: {img_path}")
            continue

        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        labelme_json = {
            'version': '5.5.0',
            'flags': {},
            'shapes': [],
            'imagePath': img_name,
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w,
        }

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue  # 至少需要1类 + 1个点

                cls_id = int(parts[0])
                label = class_names[cls_id]
                coords = list(map(float, parts[1:]))

                points = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * w
                    y = coords[i+1] * h
                    points.append([x, y])

                shape = {
                    'label': label,
                    'points': points,
                    'group_id': None,
                    'description': None,
                    'shape_type': 'polygon',
                    'flags': {},
                    'mask': None
                }
                labelme_json['shapes'].append(shape)

        out_json_path = os.path.join(output_dir, txt_name.replace(".txt", ".json"))
        with open(out_json_path, 'w', encoding='utf-8') as jf:
            json.dump(labelme_json, jf, indent=2)
        print(f"[✓] Saved: {out_json_path}")


if __name__ == '__main__':
    base_dir = "path/to/your/dataset"  # 修改为你的yolo数据集保存路径
    class_names = []
    '''
    根据你自己的数据集类别进行修改
    例如:
    class_names = ['background', 'water']
    '''

    for split in ['train', 'val', 'test']:
        txt_dir = os.path.join(base_dir, 'labels', split)
        img_dir = os.path.join(base_dir, 'images', split)
        out_dir = os.path.join(base_dir, 'labelme_json', split)
        convert_txt_to_labelme_json(txt_dir, img_dir, out_dir, class_names)
