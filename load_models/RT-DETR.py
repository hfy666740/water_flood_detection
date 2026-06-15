from typing import Optional, Tuple
from PIL import Image
from ultralytics import RTDETR
import numpy as np
import cv2

RT_DETR_LABEL_NAMES = ["bottom", "middle", "top"]

def load_rt_detr(pt_path: str, yaml_path: Optional[str] = None):
    print(f"调试信息：yaml_path={yaml_path}, pt_path={pt_path}")
    if yaml_path:
        model = RTDETR(yaml_path).load(pt_path)
    else:
        model = RTDETR(pt_path)
    try:
        names_dict = {i: n for i, n in enumerate(RT_DETR_LABEL_NAMES)}
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = names_dict
    except Exception:
        pass
    return model


def predict_rt_detr(image: Image.Image, model, conf: float = 0.25, iou: float = 0.45) -> Tuple[Image.Image, dict]:
    """
    执行推理并返回：
      - 可视化后的 PIL.Image
      - 统计数据字典：{"b_cnt": int, "m_cnt": int, "t_cnt": int, "avg_conf": str, "max_conf": str}
    
    Args:
        image: 输入图像
        model: RT-DETR模型
        conf: 置信度阈值
        iou: NMS的IoU阈值
    """
    results = model.predict(
        source=image,
        device='0',
        save=False,
        conf=conf,  # 使用传入的参数
        iou=iou,    # 使用传入的参数
        show=False,
        verbose=False
    )
    result = results[0]
    try:
        if hasattr(model, "model") and hasattr(model.model, "names"):
            result.names = model.model.names
    except Exception:
        pass

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    # if isinstance(annotated, np.ndarray):
    #     annotated = Image.fromarray(annotated)
    
    b_cnt = m_cnt = t_cnt = 0
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "cls", None) is not None:
        cls_ids = result.boxes.cls.int().cpu().numpy()
        binc = np.bincount(cls_ids, minlength=3)
        b_cnt, m_cnt, t_cnt = int(binc[0]), int(binc[1]), int(binc[2])

    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        avg_conf = f"{np.mean(confs):.2%}"
        max_conf = f"{np.max(confs):.2%}"
    else:
        avg_conf = "-"
        max_conf = "-"
    
    stats_dict = {
        "b_cnt": b_cnt,
        "m_cnt": m_cnt,
        "t_cnt": t_cnt,
        "avg_conf": avg_conf,
        "max_conf": max_conf
    }

    return annotated, stats_dict


def load_model(pt_path: str, yaml_path: Optional[str] = None):
    return load_rt_detr(pt_path, yaml_path=yaml_path)

def predict(image: Image.Image, model, conf: float = 0.25, iou: float = 0.45):
    """统一的预测接口"""
    return predict_rt_detr(image, model, conf=conf, iou=iou)