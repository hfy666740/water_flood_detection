import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO,RTDETR

def run_full_test():

    best_model_path = "path/to/your/best.pt"  # 修改为你训练好的模型权重路径
    data_yaml_path = "path/to/your/data.yaml"  # 修改为你的数据集yaml文件路径


    model = YOLO(best_model_path)

    print("📊 开始评估指标（mAP/Precision/Recall/F1）...")
    metrics = model.val(data=data_yaml_path, split="train")

    print(f"\n🎯 总体准确率指标:")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")

    conf_matrix = metrics.confusion_matrix.matrix  # shape (num_classes, num_classes)
    class_names = list(model.names.values())

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix.astype(int), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 22})
    plt.xlabel('True Label', fontsize=16)
    plt.ylabel('Predicted Label', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

    trace = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    overall_accuracy = trace / total
    print(f"\n📌 Overall Accuracy: {overall_accuracy:.4f}  (矩阵对角线之和 / 总和)")
    print("\n📈 每类精度指标:")
    per_class_data = {
        'Class': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for i, class_name in enumerate(class_names):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        print(f"🧱 {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    run_full_test()