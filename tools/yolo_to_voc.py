import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import xml.dom.minidom

yolo_dir = "path/to/your/yolo_dataset"  # 修改为你的yolo数据集保存路径

voc_dir = "path/to/voc_dataset"  # 修改为你想保存VOC数据集的路径

class_names = []
'''
例如：
classes = ["bottom","middle","top"]  请按照你的数据集类别顺序填写）
'''

splits = ["train", "val", "test"]

os.makedirs(os.path.join(voc_dir, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(voc_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(voc_dir, "ImageSets", "Main"), exist_ok=True)

split_ids = {s: [] for s in splits}

for split in splits:
    image_folder = os.path.join(yolo_dir, "images", split)
    label_folder = os.path.join(yolo_dir, "labels", split)

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img_id = os.path.splitext(img_file)[0]
        split_ids[split].append(img_id)

        im = Image.open(img_path)
        width, height = im.size

        if im.mode == "RGB":
            depth = 3
        elif im.mode == "L":
            depth = 1
        else:
            depth = 3

        # --- 构造 VOC 格式 XML ---
        annotation = ET.Element("annotation")

        folder_el = ET.SubElement(annotation, "folder")
        folder_el.text = os.path.basename(voc_dir)

        filename_el = ET.SubElement(annotation, "filename")
        filename_el.text = img_file

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size_el = ET.SubElement(annotation, "size")
        width_el = ET.SubElement(size_el, "width")
        width_el.text = str(width)
        height_el = ET.SubElement(size_el, "height")
        height_el.text = str(height)
        depth_el = ET.SubElement(size_el, "depth")
        depth_el.text = str(depth)

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        label_path = os.path.join(label_folder, img_id + ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == "":
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, bbox_width, bbox_height = parts
                    class_id = int(class_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    bbox_width = float(bbox_width)
                    bbox_height = float(bbox_height)

                    x_center_abs = x_center * width
                    y_center_abs = y_center * height
                    bbox_width_abs = bbox_width * width
                    bbox_height_abs = bbox_height * height

                    xmin = int(x_center_abs - bbox_width_abs / 2)
                    ymin = int(y_center_abs - bbox_height_abs / 2)
                    xmax = int(x_center_abs + bbox_width_abs / 2)
                    ymax = int(y_center_abs + bbox_height_abs / 2)

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)

                    obj = ET.SubElement(annotation, "object")
                    name = ET.SubElement(obj, "name")

                    name.text = classes[class_id] if class_id < len(classes) else str(class_id)
                    pose = ET.SubElement(obj, "pose")
                    pose.text = "Unspecified"
                    truncated = ET.SubElement(obj, "truncated")
                    truncated.text = "0"
                    difficult = ET.SubElement(obj, "difficult")
                    difficult.text = "0"

                    bndbox = ET.SubElement(obj, "bndbox")
                    xmin_el = ET.SubElement(bndbox, "xmin")
                    xmin_el.text = str(xmin)
                    ymin_el = ET.SubElement(bndbox, "ymin")
                    ymin_el.text = str(ymin)
                    xmax_el = ET.SubElement(bndbox, "xmax")
                    xmax_el.text = str(xmax)
                    ymax_el = ET.SubElement(bndbox, "ymax")
                    ymax_el.text = str(ymax)

        xml_str = ET.tostring(annotation, encoding="utf-8")
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")

        xml_output_path = os.path.join(voc_dir, "Annotations", img_id + ".xml")
        with open(xml_output_path, "w", encoding="utf-8") as xml_file:
            xml_file.write(pretty_xml)

        shutil.copy(img_path, os.path.join(voc_dir, "JPEGImages", img_file))

for split in splits:
    split_file = os.path.join(voc_dir, "ImageSets", "Main", split + ".txt")
    with open(split_file, "w", encoding="utf-8") as f:
        for img_id in split_ids[split]:
            f.write(img_id + "\n")

print("转换完成！")