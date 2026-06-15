"""快速测试YOLO模型是否能正常检测"""
from ultralytics import YOLO
import numpy as np
import cv2

# 加载模型
model_path = "detection_models/yolo(8n).pt"
print(f"=== 加载模型: {model_path} ===")
model = YOLO(model_path)
print(f"模型类型: {type(model)}")
print(f"names: {model.names}")
print(f"names长度: {len(model.names)}")

# 创建一张测试图片（纯白底+画一个矩形模拟车辆）
img = np.ones((640, 640, 3), dtype=np.uint8) * 200
# 画一个"车"形状（简单矩形）
cv2.rectangle(img, (200, 300), (400, 450), (100, 100, 200), -1)
cv2.circle(img, (250, 350), 20, (50, 50, 50), -1)
cv2.circle(img, (350, 350), 20, (50, 50, 50), -1)

print(f"\n=== 开始预测 conf=0.01 ===")
results = model.predict(source=img, device='cpu', save=False, conf=0.01, iou=0.45, verbose=True)
result = results[0]

print(f"\n=== 结果 ===")
if result.boxes is not None and len(result.boxes) > 0:
    cls_ids = result.boxes.cls.int().cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    print(f"检测到 {len(cls_ids)} 个目标")
    for i, (cid, c) in enumerate(zip(cls_ids, confs)):
        name = model.names.get(int(cid), "unknown")
        print(f"  [{i}] class={cid}({name}) conf={c:.3f}")
else:
    print("没有检测到任何目标")
    
    # 尝试更低置信度
    print(f"\n=== 尝试 conf=0.001 ===")
    results = model.predict(source=img, device='cpu', save=False, conf=0.001, iou=0.45, verbose=True)
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        print(f"检测到 {len(cls_ids)} 个目标")
        for i, (cid, c) in enumerate(zip(cls_ids, confs)):
            name = model.names.get(int(cid), "unknown")
            print(f"  [{i}] class={cid}({name}) conf={c:.3f}")
    else:
        print("依然没有检测到任何目标，模型可能有问题！")

print("\n=== 测试完成 ===")