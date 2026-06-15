import os
import time
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO


image_dir = "path/to/your/dataset/images"  # 待推理图像路径
mask_save_dir = "path/to/save/predicted/masks"  # 预测掩码保存路径
model_path = "path/to/your/best.pt"  # 训练好的分割模型权重路径
gt_mask_dir = "path/to/your/dataset/masks"  # 真实标签路径,与labelme_to_mask.py生成的掩码路径一致


NUM_CLASSES = 2
CLASS_NAMES = ['background', 'water']
IMAGE_SIZE = (256, 256)  # 分割评估时统一尺寸

os.makedirs(mask_save_dir, exist_ok=True)
model = YOLO(model_path)


def run_inference():
    print("🚀 开始分割推理生成掩码...")
    for image_filename in tqdm(os.listdir(image_dir), desc="⏳ 推理中"):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, image_filename)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        result = model(img, verbose=False)[0]
        masks = result.masks
        boxes = result.boxes

        if masks is None or boxes is None or len(masks.data) == 0:
            print(f"⚠️ 无掩码预测，跳过 {image_filename}")
            continue

        mask_array = np.zeros((h, w), dtype=np.uint8)

        for i, mask in enumerate(masks.data):
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            class_id = int(boxes.cls[i].item())
            mask_array[resized_mask == 1] = class_id + 1

        save_path = os.path.join(mask_save_dir, os.path.splitext(image_filename)[0] + ".png")
        cv2.imwrite(save_path, mask_array)

    print("✅ 所有图像已完成掩码生成！")


def load_mask(path):
    mask = Image.open(path).convert('L')
    mask = mask.resize(IMAGE_SIZE)
    return np.array(mask)


def compute_metrics(gt_masks, pred_masks, num_classes):
    all_gt, all_pred = [], []

    for gt, pred in zip(gt_masks, pred_masks):
        all_gt.append(gt.flatten())
        all_pred.append(pred.flatten())

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    cm = confusion_matrix(all_gt, all_pred, labels=list(range(num_classes)))
    print("\n📊 混淆矩阵:\n", cm)

    iou_list, pa_list = [], []
    for c in range(num_classes):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP

        union = TP + FP + FN
        total_gt = TP + FN

        iou = TP / union if union != 0 else 0
        pa = TP / total_gt if total_gt != 0 else 0

        iou_list.append(iou)
        pa_list.append(pa)

        print(f"\n🔹 类别 {c} ({CLASS_NAMES[c]})")
        print(f"  IoU : {iou:.4f}")
        print(f"  PA  : {pa:.4f}")

    miou = np.mean(iou_list)
    mpa = np.mean(pa_list)
    return miou, mpa


if __name__ == "__main__":
    total_start = time.time()
    run_inference()

    gt_mask_list, pred_mask_list = [], []
    gt_image_names = sorted(os.listdir(gt_mask_dir))

    valid_names = []
    for name in gt_image_names:
        pred_path = os.path.join(mask_save_dir, name)
        if os.path.exists(pred_path):
            valid_names.append(name)
        else:
            print(f"⚠️ 缺失预测掩码：{name}，跳过")

    print(f"\n✅ 有效评估图像数：{len(valid_names)}")

    for name in valid_names:
        gt_path = os.path.join(gt_mask_dir, name)
        pred_path = os.path.join(mask_save_dir, name)

        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)

        gt_mask_list.append(gt_mask)
        pred_mask_list.append(pred_mask)

    miou, mpa = compute_metrics(gt_mask_list, pred_mask_list, NUM_CLASSES)

    print("\n✅ 评估完成：")
    print(f"  👉 mIoU: {miou:.4f}")
    print(f"  👉 mPA : {mpa:.4f}")
    print(f"  ⏱️ 总耗时: {time.time() - total_start:.2f} 秒")