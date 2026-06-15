import gradio as gr
import os
import glob
from PIL import Image
import torch
import re
import html
from ultralytics import YOLO
import numpy as np
import importlib
from database import init_db, save_detection_result, get_all_detection_results, save_segmentation_result, get_all_segmentation_results
import cv2
import base64
import tempfile
from pathlib import Path
from datetime import datetime
# 模型配置
MODEL_CONFIG = {
    "目标检测": {
        "folder": "detection_models",
        "subcategories": {
            "YOLO模型": {
                "pattern": r"yolo\(([^)]+)\)\.pt$",
                "display_name": lambda x: f"YOLO-{x}",
                "yaml_pattern": r"yolo\(([^)]+)\)\.yaml$"
            },
            "其他模型": {
                "pattern": r"^(?!yolo\()(.+?)\.(?:pt|pth|onnx|bin|engine|trt|safetensors)$",
                "display_name": lambda x: x
            }
        }
    },
    "语义分割": {
        "folder": "segmentation_models", 
        "subcategories": {
            "YOLO模型": {
                "pattern": r"yolo\(([^)]+)\)\.pt$",
                "display_name": lambda x: f"YOLO-{x}",
                "yaml_pattern": r"yolo\(([^)]+)\)\.yaml$"
            },
            "其他模型": {
                "pattern": r"^(?!yolo\()(.+?)\.(?:pt|pth|onnx|bin|engine|trt|safetensors)$",
                "display_name": lambda x: x
            }
        }
    }
}

def get_image_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""
def generate_detection_stats_html(b_cnt, m_cnt, t_cnt, avg_conf, max_conf):
    """生成统一的检测统计 HTML - 使用图片形式展示"""
    # 获取图片的base64编码
    img1_b64 = get_image_base64("assets/bottom.png")
    img2_b64 = get_image_base64("assets/middle.png")
    img3_b64 = get_image_base64("assets/top.png")
    
    return f"""
<div id="stats">
  <div class="card">
    <div class="title">车辆淹没位置数量统计</div>
    <div class="vehicle-stats-container">
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img1_b64}" alt="Bottom" class="vehicle-image" />
        <div class="vehicle-label bottom">Bottom:({b_cnt})</div>
      </div>
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img2_b64}" alt="Middle" class="vehicle-image" />
        <div class="vehicle-label middle">middle:({m_cnt})</div>
      </div>
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img3_b64}" alt="Top" class="vehicle-image" />
        <div class="vehicle-label top">top:({t_cnt})</div>
      </div>
    </div>
  </div>
  <div class="card metric">
    <div class="title">平均置信度</div>
    <div class="value">{avg_conf}<span class="unit"></span></div>
  </div>
  <div class="card metric">
    <div class="title">最大置信度</div>
    <div class="value">{max_conf}<span class="unit"></span></div>
  </div>
</div>
"""

def save_detection_results_to_file(b_cnt, m_cnt, t_cnt, avg_conf, max_conf):
    """将检测结果保存为TXT文件并返回文件路径"""
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"detection_results_{timestamp}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("目标检测统计结果\n")
        f.write("=================\n\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("车辆淹没位置数量统计:\n")
        f.write(f"- Bottom: {b_cnt} 辆\n")
        f.write(f"- Middle: {m_cnt} 辆\n")
        f.write(f"- Top: {t_cnt} 辆\n")
        f.write(f"- 总数: {b_cnt + m_cnt + t_cnt} 辆\n\n")
        f.write("置信度统计:\n")
        f.write(f"- 平均置信度: {avg_conf}\n")
        f.write(f"- 最大置信度: {max_conf}\n")
    
    print(f"[保存] 检测结果已保存到: {file_path}")
    return file_path
    
def save_segmentation_results_to_file(background_ratio, water_ratio):
    """将语义分割结果保存为TXT文件并返回文件路径"""
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"segmentation_results_{timestamp}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("语义分割统计结果\n")
        f.write("=================\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("像素占比统计:\n")
        f.write(f"- 背景像素比例: {background_ratio}\n")
        f.write(f"- 水域像素比例: {water_ratio}\n")
    
    print(f"[保存] 分割结果已保存到: {file_path}")
    return file_path

def generate_segmentation_stats_html(background_ratio, water_ratio):
    """生成统一的语义分割统计 HTML（像素占比）"""
    return f"""
<div id="stats">
  <div class="card">
    <div class="title">语义分割像素占比统计</div>
    <div class="badges2">
      <div class="left">
        <span class="badge background">Background: {background_ratio}</span>
      </div>
      <div class="right">
        <span class="badge water">Water: {water_ratio}</span>
      </div>
    </div>
  </div>
  <div class="card metric">
    <div class="title">背景像素比例</div>
    <div class="value">{background_ratio}<span class="unit"></span></div>
  </div>
  <div class="card metric">
    <div class="title">水域像素比例</div>
    <div class="value">{water_ratio}<span class="unit"></span></div>
  </div>
</div>
"""
YOLO_DECTION_LABEL_NAMES = ["bottom", "middle", "top"]
YOLO_SEGMENTATION_LABEL_NAMES = ["water"]

from ultralytics.utils import plotting as uplot

YOLO_LABEL_NAMES = ["bottom", "middle", "top"]

class MyColors(uplot.Colors):
    def __call__(self, i: int, bgr: bool = False) -> tuple:
        bgr_map = {
            0: (0, 255, 0),   # bottom  -> 绿色(BGR)
            1: (255, 0, 0),   # middle  -> 蓝色(BGR)
            2: (0, 0, 255),   # top     -> 红色(BGR)
        }
        c = bgr_map.get(int(i) % 3, (255, 255, 255))
        return c if bgr else (c[2], c[1], c[0])  # 转为RGB

uplot.colors = MyColors()

def create_model_folders():
    for category in MODEL_CONFIG.values():
        folder = category["folder"]
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建文件夹: {folder}")

def get_file_size_mb(filepath):
    """获取文件大小（MB），保留1位小数"""
    try:
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 1)
    except:
        return 0.0

def get_size_badge_html(size_mb):
    """根据文件大小生成带颜色的HTML标签"""
    if size_mb < 50:
        bg_color = "#d4edda" 
        text_color = "#155724"  
        border_color = "#c3e6cb"  
    elif size_mb < 200:
        bg_color = "#fff3cd"  
        text_color = "#856404"  
        border_color = "#ffeaa7" 
    else:
        bg_color = "#f8d7da"  
        text_color = "#721c24" 
        border_color = "#f5c6cb"  
    
    return f"""<span style="display:inline-block; padding:2px 8px; margin-left:8px; 
                background:{bg_color}; color:{text_color}; 
                border:1px solid {border_color}; border-radius:4px; 
                font-size:12px; font-weight:600;">{size_mb}M</span>"""

def get_available_models(category, subcategory):
    if category not in MODEL_CONFIG:
        return ["无可用类别"]
    folder = MODEL_CONFIG[category]["folder"]
    if subcategory not in MODEL_CONFIG[category]["subcategories"]:
        return ["无可用子类别"]
    config = MODEL_CONFIG[category]["subcategories"][subcategory]
    pattern = config["pattern"]
    display_func = config["display_name"]
    
    if not os.path.exists(folder):
        print(f"调试信息: 文件夹不存在 {folder}")
        return ["文件夹不存在"]
    
    # 获取所有 pt 和 pth 文件
    model_files = glob.glob(os.path.join(folder, "*.pt")) + glob.glob(os.path.join(folder, "*.pth"))
    print(f"调试信息: 找到 {len(model_files)} 个模型文件")
    for f in model_files:
        print(f"  - {os.path.basename(f)}")
    
    if not model_files:
        return ["无模型文件"]
    
    available_models = []
    for model_file in model_files:
        filename = os.path.basename(model_file)
        match = re.search(pattern, filename)
        if match:
            if subcategory == "其他模型" and filename.lower().startswith("yolo("):
                continue
            model_name = match.group(1)
            display_name = display_func(model_name)
            
            # 获取文件大小
            size_mb = get_file_size_mb(model_file)
            
            model_info = {
                "display": display_name,
                "display_with_size": f"{display_name} ({size_mb}M)",
                "filepath": model_file,
                "filename": filename,
                "model_name": model_name,
                "size_mb": size_mb  
            }
            if subcategory == "YOLO模型":
                yaml_pattern = config["yaml_pattern"]
                yaml_filename = f"yolo({model_name}).yaml"
                yaml_filepath = os.path.join(folder, yaml_filename)
                
                if os.path.exists(yaml_filepath):
                    model_info["yaml_filepath"] = yaml_filepath
                    model_info["yaml_filename"] = yaml_filename
                else:
                    # 如果yaml不存在,允许直接使用.pt权重(沿用模型自身的类别数)
                    print(f"提示: {folder} 中的模型 {model_name} 没有YAML文件,将直接加载.pt权重")
            
            available_models.append(model_info)
        else:
            print(f"调试信息: 文件 {filename} 不匹配模式 {pattern}")
    
    print(f"调试信息: 最终找到 {len(available_models)} 个可用模型")
    if not available_models:
        return ["无匹配模型"]
    available_models.sort(key=lambda x: x["display"])
    return available_models

# 模型加载函数
def _resolve_other_model_adapter(model_name: str, adapter_dir: str = "load_models"):
    """
    根据模型名解析对应的 load/predict 函数
    增强版：支持连字符、下划线的灵活匹配
    
    Args:
        model_name: 模型名称
        adapter_dir: 适配器所在文件夹，默认 "load_models"，分割模型可用 "load_seg_models"
    """
    if not os.path.exists(adapter_dir):
        raise RuntimeError(f"{adapter_dir} 目录不存在")
    
    normalized_name = model_name.lower().replace(' ', '_')
    
    available_files = [f[:-3] for f in os.listdir(adapter_dir)
                       if f.endswith('.py') and not f.startswith('_')]
    
    def fuzzy_match(name, target):
        """灵活匹配：忽略连字符、下划线、大小写差异"""
        name_clean = name.lower().replace('-', '').replace('_', '')
        target_clean = target.lower().replace('-', '').replace('_', '')
        return name_clean == target_clean
    
    matched_module = None
    for file in available_files:
        if fuzzy_match(model_name, file):
            matched_module = file
            break
    
    if not matched_module:
        raise RuntimeError(
            f"找不到模型适配器: {model_name}\n"
            f"可用模块: {available_files}\n"
            f"提示: 模型文件名应与 {adapter_dir}/ 下的 .py 文件对应"
        )
    
    module_path = f"{adapter_dir.replace('/', '.').replace(os.sep, '.')}.{matched_module}"
    
    import sys
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        module = importlib.import_module(f"{adapter_dir}.{matched_module}")
    
    safe_name = normalized_name.replace('-', '_')
    load_fn = (
        getattr(module, f"load_{safe_name}", None) or
        getattr(module, "load_model", None)
    )
    predict_fn = (
        getattr(module, f"predict_{safe_name}", None) or
        getattr(module, "predict", None)
    )
    
    if not load_fn or not predict_fn:
        raise RuntimeError(
            f"模型适配器 {matched_module}.py 缺少必要函数\n"
            f"需要: load_model() 和 predict() 或 load_{safe_name}() 和 predict_{safe_name}()"
        )
    
    return load_fn, predict_fn

def load_model(model_info, subcategory, task_type="detection"):
    """加载模型 - 根据模型类型使用不同的加载方式"""
    if isinstance(model_info, str) and ("无" in model_info or "不存在" in model_info):
        return f"状态: {model_info}", None
    
    try:
        model_name = model_info['display']
        
        if subcategory == "YOLO模型":
            pt_path = model_info['filepath']
            if task_type == "segmentation":
                model = YOLO(pt_path) 
                
                actual_nc = getattr(model.model, 'nc', 1) if hasattr(model, 'model') else 1
                
                if actual_nc == 1:
                    names_dict = {0: "water"} 
                elif actual_nc == 2:
                    names_dict = {0: "water", 1: "other"} 
                else:
                    names_dict = {i: f"class_{i}" for i in range(actual_nc)}
                
                try:
                    if hasattr(model, "model") and hasattr(model.model, "names"):
                        model.model.names = names_dict
                except Exception:
                    pass
                
                return f"状态: YOLO分割模型 {model_name} 加载成功\n权重文件: {os.path.basename(pt_path)}\n类别数: {actual_nc}", model
            
            else:
                yaml_path = model_info.get('yaml_filepath')
                if yaml_path and os.path.exists(yaml_path):
                    model = YOLO(yaml_path).load(pt_path)
                    names_dict = {i: n for i, n in enumerate(YOLO_DECTION_LABEL_NAMES)}

                    try:
                        if hasattr(model, "model") and hasattr(model.model, "names"):
                            model.model.names = names_dict
                    except Exception:
                        pass

                    return f"状态: YOLO检测模型 {model_name} 加载成功\n配置文件: {os.path.basename(yaml_path)}\n权重文件: {os.path.basename(pt_path)}", model
                else:
                    # 无yaml,直接加载.pt权重(沿用权重自身的类别数和names)
                    model = YOLO(pt_path)
                    actual_nc = getattr(getattr(model, 'model', None), 'nc', 80) or 80
                    return f"状态: YOLO检测模型 {model_name} 加载成功(仅权重)\n权重文件: {os.path.basename(pt_path)}\n类别数: {actual_nc}", model
        
        else:
            pt_path = model_info['filepath']
            mname = model_info.get('model_name') or os.path.splitext(os.path.basename(pt_path))[0]
            
            adapter_dir = "load_seg_models" if task_type == "segmentation" else "load_models"
            load_fn, predict_fn = _resolve_other_model_adapter(mname, adapter_dir=adapter_dir)
            
            model = load_fn(pt_path)
            bundle = {"model": model, "predict_fn": predict_fn, "name": mname}
            
            task_name = "分割" if task_type == "segmentation" else "检测"
            return f"状态: 其他{task_name}模型 {model_name} 加载成功", bundle
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"状态: 模型加载失败 - {str(e)}", None

def process_yolo_detection(image, model_info, loaded_model, conf=0.01, iou=0.45):
    if image is None:
        return None, "请先上传图像"
    if loaded_model is None:
        return None, "请先加载YOLO模型"

    try:
        # === 1. 统一输入为RGB numpy数组 ===
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    img_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    img_np = image.copy()
            else:
                img_np = image
        else:
            return None, f"不支持的图像类型: {type(image)}"

        img_h, img_w = img_np.shape[:2]

        # === 2. CPU推理(极低阈值确保有输出) ===
        results = loaded_model.predict(
            source=img_np,
            device='cpu',
            save=False,
            conf=max(conf, 0.001),
            iou=iou,
            show=False,
            verbose=True,
            max_det=100
        )
        result = results[0]

        # === 3. 获取模型信息 ===
        model_names = loaded_model.names if hasattr(loaded_model, 'names') else {}
        is_coco_model = len(model_names) >= 80
        COCO_VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

        b_cnt = m_cnt = t_cnt = 0
        vehicle_confs = []
        all_confs = []          # 所有目标的置信度(用于显示)
        all_detections_list = [] # 用于调试表格
        draw_boxes = []

        has_boxes = (getattr(result, "boxes", None) is not None and
                     getattr(result.boxes, "cls", None) is not None and
                     len(result.boxes) > 0)

        if has_boxes:
            cls_ids = result.boxes.cls.int().cpu().numpy()
            confs_all = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None

            for idx in range(len(cls_ids)):
                cid = int(cls_ids[idx])
                c = float(confs_all[idx])
                label_name = model_names.get(cid, f"class_{cid}")
                all_confs.append(c)

                x1, y1, x2, y2 = (int(xyxy[idx][0]), int(xyxy[idx][1]),
                                   int(xyxy[idx][2]), int(xyxy[idx][3]))

                is_vehicle = is_coco_model and cid in COCO_VEHICLE_IDS

                # ===== 统一蓝色粗框(BGR: 255,0,0 = 蓝色) =====
                BLUE = (255, 0, 0)
                GREEN = (0, 255, 0)
                YELLOW = (0, 255, 255)
                RED = (0, 0, 255)

                if is_vehicle:
                    # 车辆类 - 根据位置判断淹没程度
                    box_center_y = (y1 + y2) / 2.0
                    ratio = box_center_y / img_h
                    vehicle_confs.append(c)
                    if ratio > 0.65:
                        flood_label = "Vehicle-Bottom"
                        box_color = GREEN
                        b_cnt += 1
                    elif ratio > 0.35:
                        flood_label = "Vehicle-Middle"
                        box_color = YELLOW
                        m_cnt += 1
                    else:
                        flood_label = "Vehicle-Top"
                        box_color = RED
                        t_cnt += 1
                    display_label = f"{flood_label} {c:.0%}"
                else:
                    # 非车辆类 - 也用蓝色框标出
                    box_color = BLUE
                    display_label = f"{label_name} {c:.0%}"

                draw_boxes.append((x1, y1, x2, y2, display_label, box_color))
                all_detections_list.append((label_name, c, "Vehicle" if is_vehicle else "Other"))

        # === 4. 在图上绘制检测框（全部蓝色系） ===
        annotated = img_np.copy()
        for (x1, y1, x2, y2, label, color) in draw_boxes:
            # 粗蓝色边框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            # 标签背景 + 文字
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(y1 - th - 10, 0)
            cv2.rectangle(annotated, (x1, label_y), (x1 + tw + 8, label_y + th + 10), color, -1)
            cv2.putText(annotated, label, (x1 + 4, label_y + th + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 如果没有任何检测框，在图上写提示
        if not draw_boxes:
            cv2.putText(annotated, "No objects detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated, f"conf={conf}, model={len(model_names)} classes",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # === 5. 计算统计数据 ===
        total_detected = len(draw_boxes)
        if all_confs:
            avg_conf = f"{np.mean(all_confs):.2%}"
            max_conf = f"{np.max(all_confs):.2%}"
        else:
            avg_conf = "0.00%"
            max_conf = "0.00%"

        stats_html = generate_detection_stats_html(b_cnt, m_cnt, t_cnt, avg_conf, max_conf)

        # 调试详情面板
        detail_html = f"""
<div style="margin-top:12px;padding:14px;background:#f0f4ff;border-radius:8px;border-left:4px solid #3F6BC8;">
  <p style="margin:0 0 8px 0;font-size:13px;">
    <b>检测信息</b> | 模型: {len(model_names)}类{'(COCO)' if is_coco_model else '(专用)'} |
    阈值: {conf} | 总检出: <b style="color:#3F6BC8;">{total_detected}</b>个目标 |
    车辆: <b style="color:green;">{b_cnt+m_cnt+t_cnt}</b>辆
  </p>"""

        if all_detections_list:
            detail_html += """<table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:6px;">
  <tr style="background:#e0e7ff;"><th style="padding:4px 8px;border:1px solid #ccc;">#</th>
  <th style="padding:4px 8px;border:1px solid #ccc;">类别</th><th style="padding:4px 8px;border:1px solid #ccc;">置信度</th>
  <th style="padding:4px 8px;border:1px solid #ccc;">类型</th></tr>"""
            for i, (name, c, typ) in enumerate(all_detections_list):
                bg = "#e8f5e9" if typ == "Vehicle" else "#fff"
                detail_html += f"<tr style='background:{bg};'><td>{i+1}</td><td>{name}</td><td>{c:.1%}</td><td>{typ}</td></tr>"
            detail_html += "</table>"
        else:
            detail_html += """<p style="margin:6px 0 0 0;color:#999;font-size:12px;">
  未检测到目标。COCO预训练模型对水淹场景检测能力有限。<br>
  建议：使用针对水淹车辆训练的专用模型替换 detection_models/ 中的权重文件。</p>"""

        detail_html += "</div>"
        stats_html += detail_html

        # === 6. 保存到SQLite数据库 YOLO_flooded ===
        try:
            orig_bytes = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1].tobytes()
            save_detection_result(
                image_name=f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                image_bytes=orig_bytes,
                b_cnt=b_cnt, m_cnt=m_cnt, t_cnt=t_cnt,
                avg_conf=avg_conf, max_conf=max_conf
            )
        except Exception as dbe:
            print(f"[DB] save failed: {dbe}")

        return annotated, stats_html

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"YOLO处理失败: {str(e)}"
def process_yolo_detection_video(video_file, model, conf=0.05, iou=0.45, progress=gr.Progress()):
    """
    使用YOLO模型处理视频文件
    """
    if video_file is None:
        return None, "请先上传视频"
    
    if model is None:
        return None, "请先加载YOLO模型"
    
    try:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"[视频] 使用设备: {device}")

        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "无法打开视频文件"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        codecs_to_try = [
            ('XVID', 'avi'), 
            ('MJPG', 'avi'),  
            ('mp4v', 'mp4'),  
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                test_out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    test_out.release()
                    os.unlink(test_path)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"[视频] 使用编码器: {codec}")
                    break
            except:
                continue
        
        if out is None or not out.isOpened():
            cap.release()
            return None, "无法创建输出视频文件，请安装FFmpeg或检查OpenCV配置"
        
        total_b_cnt = 0
        total_m_cnt = 0
        total_t_cnt = 0
        all_confs = []
        
        frame_count = 0
        
        # 判断是否为COCO预训练模型
        model_nc = getattr(getattr(model, "model", None), "nc", 0)
        is_coco_model = (model_nc >= 80)
        COCO_VEHICLE_IDS = {2, 3, 5, 7}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if total_frames > 0 and callable(progress):
                progress(frame_count / total_frames, f"处理中: {frame_count}/{total_frames} 帧")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model.predict(
                source=frame_rgb,
                device=device,
                save=False,
                conf=conf,
                iou=iou,
                show=False,
                verbose=False
            )
            
            result = results[0]
            img_h = frame_rgb.shape[0]
            draw_boxes = []

            if getattr(result, "boxes", None) is not None and getattr(result.boxes, "cls", None) is not None:
                cls_ids = result.boxes.cls.int().cpu().numpy()
                confs_all = result.boxes.conf.cpu().numpy()
                xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None

                if is_coco_model:
                    vehicle_mask = np.isin(cls_ids, list(COCO_VEHICLE_IDS))
                    vehicle_indices = np.where(vehicle_mask)[0]
                    for idx in vehicle_indices:
                        c = float(confs_all[idx])
                        all_confs.append(c)
                        if xyxy is not None and img_h > 0:
                            x1, y1, x2, y2 = xyxy[idx]
                            box_center_y = (y1 + y2) / 2.0
                            ratio = box_center_y / img_h
                            if ratio > 0.65:
                                label = f"Vehicle-Bottom {c:.2f}"
                                color = (0, 255, 0)
                                total_b_cnt += 1
                            elif ratio > 0.35:
                                label = f"Vehicle-Middle {c:.2f}"
                                color = (0, 255, 255)
                                total_m_cnt += 1
                            else:
                                label = f"Vehicle-Top {c:.2f}"
                                color = (0, 0, 255)
                                total_t_cnt += 1
                            draw_boxes.append((int(x1), int(y1), int(x2), int(y2), label, color))
                        else:
                            total_m_cnt += 1
                            if xyxy is not None:
                                x1, y1, x2, y2 = xyxy[idx]
                                draw_boxes.append((int(x1), int(y1), int(x2), int(y2), f"Vehicle {c:.2f}", (255, 0, 0)))
                else:
                    binc = np.bincount(cls_ids, minlength=3)
                    total_b_cnt += int(binc[0])
                    total_m_cnt += int(binc[1])
                    total_t_cnt += int(binc[2])
                    all_confs.extend(confs_all.tolist())
                    if xyxy is not None:
                        for idx in range(len(cls_ids)):
                            cid = int(cls_ids[idx])
                            x1, y1, x2, y2 = xyxy[idx]
                            label_name = YOLO_DECTION_LABEL_NAMES[cid] if cid < len(YOLO_DECTION_LABEL_NAMES) else f"class_{cid}"
                            c = float(confs_all[idx])
                            color = (255, 0, 0) if cid == 0 else (0, 255, 255) if cid == 1 else (0, 0, 255)
                            draw_boxes.append((int(x1), int(y1), int(x2), int(y2), f"{label_name} {c:.2f}", color))

            # 手动绘制检测框
            annotated_frame = frame.copy()
            for (x1, y1, x2, y2, label, color) in draw_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(annotated_frame)
        
        cap.release()
        out.release()

        if len(all_confs) > 0:
            avg_conf = f"{np.mean(all_confs):.2%}"
            max_conf = f"{np.max(all_confs):.2%}"
        else:
            avg_conf = "-"
            max_conf = "-"
        
        stats_html = f"""
        <div style="margin-bottom: 16px; padding: 12px; background: #e8f5e8; border-radius: 6px;">
            <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 视频处理完成</h4>
            <p style="margin: 4px 0;"><strong>总帧数:</strong> {frame_count}</p>
            <p style="margin: 4px 0;"><strong>视频时长:</strong> {frame_count/fps:.2f}秒</p>
            <p style="margin: 4px 0;"><strong>检测到的总目标数:</strong> {total_b_cnt + total_m_cnt + total_t_cnt}</p>
        </div>
        """ + generate_detection_stats_html(total_b_cnt, total_m_cnt, total_t_cnt, avg_conf, max_conf)
        
        results_file = save_detection_results_to_file(total_b_cnt, total_m_cnt, total_t_cnt, avg_conf, max_conf)
        
        return output_path, stats_html, results_file
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"视频处理失败: {str(e)}", None

def process_other_detection(image, model_info, loaded_model, conf=0.25, iou=0.45):
    """其他模型目标检测处理函数"""
    if image is None:
        return None, "请先上传图像"
    
    if loaded_model is None:
        return None, "请先加载模型"
    
    try:
        if isinstance(loaded_model, dict) and "predict_fn" in loaded_model and "model" in loaded_model:
            predict_fn = loaded_model["predict_fn"]
            model = loaded_model["model"]

            predicted_image, result_data = predict_fn(image, model, conf=conf, iou=iou)
            
            if isinstance(result_data, str) and "<div" in result_data:
                return predicted_image, result_data
            elif isinstance(result_data, dict):
                html = generate_detection_stats_html(
                    result_data.get("b_cnt", 0),
                    result_data.get("m_cnt", 0),
                    result_data.get("t_cnt", 0),
                    result_data.get("avg_conf", "-"),
                    result_data.get("max_conf", "-")
                )
                return predicted_image, html
            else:
                return predicted_image, generate_detection_stats_html(0, 0, 0, "-", "-")
        else:
            # 回退逻辑
            processed_image = image.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(processed_image)
            width, height = processed_image.size
            draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                          outline="blue", width=3)
            draw.text((width//4 + 10, height//4 + 10), "其他模型检测", fill="blue")
            return processed_image, generate_detection_stats_html(0, 0, 0, "-", "-")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"其他模型处理失败: {str(e)}"

def process_other_detection_video(video_file, loaded_model, conf=0.25, iou=0.45, progress=gr.Progress()):
    """
    使用其他模型处理视频文件进行目标检测
    Args:
        video_file: 上传的视频文件路径
        loaded_model: 加载的模型bundle (dict with "model" and "predict_fn")
        conf: 置信度阈值
        iou: IoU阈值
        progress: Gradio进度条
    
    Returns:
        处理后的视频路径, 统计信息HTML, 结果文件路径
    """
    if video_file is None:
        return None, "请先上传视频", None
    
    if loaded_model is None:
        return None, "请先加载模型", None
    
    if not isinstance(loaded_model, dict) or "predict_fn" not in loaded_model or "model" not in loaded_model:
        return None, "模型格式错误，请重新加载模型", None
    
    try:
        predict_fn = loaded_model["predict_fn"]
        model = loaded_model["model"]
        
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "无法打开视频文件", None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        codecs_to_try = [
            ('XVID', 'avi'), 
            ('MJPG', 'avi'),  
            ('mp4v', 'mp4'),  
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                test_out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    test_out.release()
                    os.unlink(test_path)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"使用编码器: {codec}")
                    break
            except:
                continue
        
        if out is None or not out.isOpened():
            cap.release()
            return None, "无法创建输出视频文件，请安装FFmpeg或检查OpenCV配置", None
        
        total_b_cnt = 0
        total_m_cnt = 0
        total_t_cnt = 0
        all_confs = []
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if total_frames > 0 and callable(progress):
                progress(frame_count / total_frames, f"处理中: {frame_count}/{total_frames} 帧")
            
            # 将BGR转换为RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 使用模型的predict函数处理单帧
            predicted_image, result_data = predict_fn(frame_pil, model, conf=conf, iou=iou)
            
            # 将PIL Image转换为numpy array，然后转换为BGR用于OpenCV
            if isinstance(predicted_image, Image.Image):
                annotated_frame = np.array(predicted_image)
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            elif isinstance(predicted_image, np.ndarray):
                annotated_frame = predicted_image
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            else:
                annotated_frame = frame
            
            # 收集统计数据
            if isinstance(result_data, dict):
                total_b_cnt += result_data.get("b_cnt", 0)
                total_m_cnt += result_data.get("m_cnt", 0)
                total_t_cnt += result_data.get("t_cnt", 0)
                # 注意：置信度信息可能不在result_data中，需要从predict函数返回
            elif isinstance(result_data, str) and "<div" in result_data:
                # 如果是HTML字符串，尝试解析统计数据
                import re
                b_match = re.search(r'Bottom:\((\d+)\)', result_data)
                m_match = re.search(r'middle:\((\d+)\)', result_data)
                t_match = re.search(r'top:\((\d+)\)', result_data)
                if b_match:
                    total_b_cnt += int(b_match.group(1))
                if m_match:
                    total_m_cnt += int(m_match.group(1))
                if t_match:
                    total_t_cnt += int(t_match.group(1))
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        # 计算平均置信度和最大置信度
        # 注意：其他模型可能不返回置信度信息，这里使用"-"
        avg_conf = "-"
        max_conf = "-"
        
        stats_html = f"""
        <div style="margin-bottom: 16px; padding: 12px; background: #e8f5e8; border-radius: 6px;">
            <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 视频处理完成</h4>
            <p style="margin: 4px 0;"><strong>总帧数:</strong> {frame_count}</p>
            <p style="margin: 4px 0;"><strong>视频时长:</strong> {frame_count/fps:.2f}秒</p>
            <p style="margin: 4px 0;"><strong>检测到的总目标数:</strong> {total_b_cnt + total_m_cnt + total_t_cnt}</p>
        </div>
        """ + generate_detection_stats_html(total_b_cnt, total_m_cnt, total_t_cnt, avg_conf, max_conf)
        
        results_file = save_detection_results_to_file(total_b_cnt, total_m_cnt, total_t_cnt, avg_conf, max_conf)
        
        return output_path, stats_html, results_file
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"视频处理失败: {str(e)}", None
# YOLO分割处理函数
def process_yolo_segmentation(image, model_info, loaded_model, conf=0.25, iou=0.45):
    if image is None:
        return None, "请先上传图像"
    if loaded_model is None:
        return None, "请先加载YOLO分割模型"
    
    try:
        results = loaded_model.predict(
            source=image,
            device='0',
            save=False,
            conf=conf,  
            iou=iou,   
            show=False,
            verbose=False
        )
        
        result = results[0]
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        from PIL import Image
        import numpy as np
        if isinstance(annotated_image, np.ndarray):
            annotated_image = Image.fromarray(annotated_image)
        
        background_ratio = "-"
        water_ratio = "-"
        
        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()  # shape: [N, H, W]
            
            h, w = masks.shape[1], masks.shape[2]
            final_mask = np.zeros((h, w), dtype=np.float32)
            
            for mask in masks:
                final_mask = np.maximum(final_mask, mask)
            water_mask = (final_mask > 0.5).astype(np.uint8)
            
            total_pixels = h * w
            water_pixels = np.sum(water_mask)
            bg_pixels = total_pixels - water_pixels
            
            background_ratio = f"{(bg_pixels / total_pixels):.2%}"
            water_ratio = f"{(water_pixels / total_pixels):.2%}"

        stats_html = generate_segmentation_stats_html(background_ratio, water_ratio)
        
        return annotated_image, stats_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"YOLO分割处理失败: {str(e)}"

def process_yolo_segmentation_video(video_file, model, conf=0.25, iou=0.45, progress=gr.Progress()):
    """
    使用YOLO模型处理视频文件进行语义分割
    Args:
        video_file: 上传的视频文件路径
        model: 加载的YOLO分割模型
        conf: 置信度阈值
        iou: IoU阈值
        progress: Gradio进度条
    
    Returns:
        处理后的视频路径, 统计信息HTML
    """
    if video_file is None:
        return None, "请先上传视频"
    
    if model is None:
        return None, "请先加载YOLO分割模型"
    
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, "无法打开视频文件"
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name

        codecs_to_try = [
            ('XVID', 'avi'), 
            ('MJPG', 'avi'),  
            ('mp4v', 'mp4'),  
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                test_out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    test_out.release()
                    os.unlink(test_path)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"使用编码器: {codec}")
                    break
            except:
                continue
        
        if out is None or not out.isOpened():
            cap.release()
            return None, "无法创建输出视频文件，请安装FFmpeg或检查OpenCV配置"
        
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if total_frames > 0 and callable(progress):
                progress(frame_count / total_frames, f"分割处理中: {frame_count}/{total_frames} 帧")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model.predict(
                source=frame_rgb,
                device='0',
                save=False,
                conf=conf,
                iou=iou,
                show=False,
                verbose=False
            )
            
            result = results[0]
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        stats_html = f"""
        <div style="margin-bottom: 16px; padding: 12px; background: #e8f5e8; border-radius: 6px;">
            <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 视频分割完成</h4>
            <p style="margin: 4px 0;"><strong>总帧数:</strong> {frame_count}</p>
            <p style="margin: 4px 0;"><strong>视频时长:</strong> {frame_count/fps:.2f}秒</p>
        </div>
        """
        
        return output_path, stats_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"视频分割处理失败: {str(e)}"

def process_other_segmentation(image, model_info, loaded_model):
    """其他模型语义分割处理函数"""
    if image is None:
        return None, "请先上传图像"
    
    if loaded_model is None:
        return None, "请先加载模型"
    
    try:
        if isinstance(loaded_model, dict) and "predict_fn" in loaded_model and "model" in loaded_model:
            predict_fn = loaded_model["predict_fn"]
            model = loaded_model["model"]
            predicted_image, result_data = predict_fn(image, model)

            if isinstance(result_data, str) and "<div" in result_data:
                return predicted_image, result_data
            elif isinstance(result_data, dict):
                html = generate_segmentation_stats_html(
                    result_data.get("background_ratio", "-"),
                    result_data.get("water_ratio", "-")
                )
                return predicted_image, html
            else:
                return predicted_image, generate_segmentation_stats_html("-", "-")
        else:
            processed_image = image.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(processed_image)
            width, height = processed_image.size
            draw.ellipse([width//3, height//3, 2*width//3, 2*height//3], 
                        fill=None, outline="green", width=3)
            draw.text((width//3 + 10, height//3 + 10), "其他模型分割", fill="green")
            return processed_image, generate_segmentation_stats_html("-", "-")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"其他模型分割处理失败: {str(e)}"

def process_other_segmentation_video(video_file, loaded_model, conf=0.25, iou=0.45, progress=gr.Progress()):
    """
    使用其他模型处理视频文件进行语义分割
    Args:
        video_file: 上传的视频文件路径
        loaded_model: 加载的模型bundle (dict with "model" and "predict_fn")
        conf: 置信度阈值（某些模型可能不使用）
        iou: IoU阈值（某些模型可能不使用）
        progress: Gradio进度条
    
    Returns:
        处理后的视频路径, 统计信息HTML
    """
    if video_file is None:
        return None, "请先上传视频"
    
    if loaded_model is None:
        return None, "请先加载模型"
    
    if not isinstance(loaded_model, dict) or "predict_fn" not in loaded_model or "model" not in loaded_model:
        return None, "模型格式错误，请重新加载模型"
    
    try:
        predict_fn = loaded_model["predict_fn"]
        model = loaded_model["model"]
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, "无法打开视频文件"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name

        codecs_to_try = [
            ('XVID', 'avi'), 
            ('MJPG', 'avi'),  
            ('mp4v', 'mp4'),  
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                test_out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    test_out.release()
                    os.unlink(test_path)
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}').name
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"使用编码器: {codec}")
                    break
            except:
                continue
        
        if out is None or not out.isOpened():
            cap.release()
            return None, "无法创建输出视频文件，请安装FFmpeg或检查OpenCV配置"
        
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if total_frames > 0 and callable(progress):
                progress(frame_count / total_frames, f"分割处理中: {frame_count}/{total_frames} 帧")

            # 将BGR转换为RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # 使用模型的predict函数处理单帧
            # 注意：分割模型的predict函数可能不接受conf和iou参数
            try:
                # 尝试传递device参数（某些分割模型需要）
                predicted_image, result_data = predict_fn(frame_pil, model, device='0')
            except TypeError:
                # 如果predict函数不接受device参数，尝试不传参数
                try:
                    predicted_image, result_data = predict_fn(frame_pil, model)
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {e}")
                    predicted_image = frame_pil
                    result_data = {"background_ratio": "-", "water_ratio": "-"}
            
            # 将PIL Image转换为numpy array，然后转换为BGR用于OpenCV
            if isinstance(predicted_image, Image.Image):
                annotated_frame = np.array(predicted_image)
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            elif isinstance(predicted_image, np.ndarray):
                annotated_frame = predicted_image
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        stats_html = f"""
        <div style="margin-bottom: 16px; padding: 12px; background: #e8f5e8; border-radius: 6px;">
            <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 视频分割完成</h4>
            <p style="margin: 4px 0;"><strong>总帧数:</strong> {frame_count}</p>
            <p style="margin: 4px 0;"><strong>视频时长:</strong> {frame_count/fps:.2f}秒</p>
        </div>
        """
        
        return output_path, stats_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"视频分割处理失败: {str(e)}"

def process_combined_analysis(image, 
                              det_subcategory, det_model_display, det_models, det_model_obj, det_conf, det_iou,
                              seg_subcategory, seg_model_display, seg_models, seg_model_obj, seg_conf, seg_iou):
    """联合分析处理函数：分别对原图执行检测和分割"""
    if image is None:
        return None, None, "请先上传图像", "请先上传图像", None, gr.update(visible=False)
    
    if det_model_obj is None:
        return None, None, "请先加载检测模型", "等待检测完成", None, gr.update(visible=False)
    
    if seg_model_obj is None:
        return None, None, "检测模型已加载，但分割模型未加载", "请先加载分割模型", None, gr.update(visible=False)
    
    try:
        det_model_info = get_current_model_info("目标检测", det_subcategory, det_model_display, det_models)
        
        if det_subcategory == "YOLO模型":
            detection_result, detection_stats = process_yolo_detection(image, det_model_info, det_model_obj, det_conf, det_iou)
        else:
            detection_result, detection_stats = process_other_detection(image, det_model_info, det_model_obj, det_conf, det_iou)
        
        seg_model_info = get_current_model_info("语义分割", seg_subcategory, seg_model_display, seg_models)
        
        if seg_subcategory == "YOLO模型":
            segmentation_result, segmentation_stats = process_yolo_segmentation(image, seg_model_info, seg_model_obj, seg_conf, seg_iou)
        else:
            segmentation_result, segmentation_stats = process_other_segmentation(image, seg_model_info, seg_model_obj)
        
        combined_file = None
        if detection_result is not None and segmentation_result is not None:
            combined_file = save_combined_results_to_file(detection_stats, segmentation_stats)
        
        return detection_result, segmentation_result, detection_stats, segmentation_stats, combined_file, gr.update(visible=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"联合分析失败: {str(e)}", "分析中断", None, gr.update(visible=False)

# 视频联合分析处理函数
def process_combined_video_analysis(video,
                                   det_subcategory, det_model_display, det_models, det_model_obj, det_conf, det_iou,
                                   seg_subcategory, seg_model_display, seg_models, seg_model_obj, seg_conf, seg_iou,
                                   progress=gr.Progress()):
    if video is None:
        yield None, None, "请先上传视频", "请先上传视频"
        return
    
    if det_model_obj is None:
        yield None, None, "请先加载检测模型", "等待检测完成"
        return
    
    if seg_model_obj is None:
        yield None, None, "检测模型已加载，但分割模型未加载", "请先加载分割模型"
        return
    
    try:
        # 阶段1：目标检测处理（使用第一个进度条）
        progress(0.0, desc="🎯 阶段1/2: 开始目标检测处理...")
        
        # 创建检测阶段的进度回调函数
        def detection_progress_callback(current_progress, desc=""):
            # 将检测进度映射到总进度的前50%
            total_progress = current_progress * 0.5
            progress(total_progress, desc=f"🎯 目标检测: {desc}")
        
        # 根据模型类型选择相应的处理函数
        if det_subcategory == "YOLO模型":
            detection_video, detection_stats, _ = process_yolo_detection_video(
                video, det_model_obj, det_conf, det_iou, progress=detection_progress_callback
            )
        else:
            # 其他模型
            detection_video, detection_stats, _ = process_other_detection_video(
                video, det_model_obj, det_conf, det_iou, progress=detection_progress_callback
            )

        # 检测完成，立即显示结果
        progress(0.5, desc="✅ 目标检测完成，开始语义分割...")
        yield detection_video, None, detection_stats, "<p style='text-align:center; color:#666;'>🎨 正在进行语义分割处理...</p>"
        
        # 阶段2：语义分割处理（使用第二个进度条）
        # 创建分割阶段的进度回调函数
        def segmentation_progress_callback(current_progress, desc=""):
            # 将分割进度映射到总进度的后50%
            total_progress = 0.5 + (current_progress * 0.5)
            progress(total_progress, desc=f"🎨 语义分割: {desc}")
        
        # 根据模型类型选择相应的处理函数
        if seg_subcategory == "YOLO模型":
            segmentation_video, segmentation_stats = process_yolo_segmentation_video(
                video, seg_model_obj, seg_conf, seg_iou, progress=segmentation_progress_callback
            )
        else:
            # 其他模型
            segmentation_video, segmentation_stats = process_other_segmentation_video(
                video, seg_model_obj, seg_conf, seg_iou, progress=segmentation_progress_callback
            )
        
        # 所有处理完成
        progress(1.0, desc="🎉 联合分析完成！")
        yield detection_video, segmentation_video, detection_stats, segmentation_stats
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, None, f"视频联合分析失败: {str(e)}", "分析中断"

def save_combined_results_to_file(detection_stats, segmentation_stats):
    import os
    from datetime import datetime
    import re
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"combined_analysis_{timestamp}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("联合分析结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("【目标检测结果】\n")
        f.write("-" * 50 + "\n")
        
        b_cnt_match = re.search(r'Bottom:\((\d+)\)', detection_stats)
        m_cnt_match = re.search(r'middle:\((\d+)\)', detection_stats)
        t_cnt_match = re.search(r'top:\((\d+)\)', detection_stats)
        
        if b_cnt_match and m_cnt_match and t_cnt_match:
            b_cnt = int(b_cnt_match.group(1))
            m_cnt = int(m_cnt_match.group(1))
            t_cnt = int(t_cnt_match.group(1))
            f.write(f"- Bottom: {b_cnt} 辆\n")
            f.write(f"- Middle: {m_cnt} 辆\n")
            f.write(f"- Top: {t_cnt} 辆\n")
            f.write(f"- 总数: {b_cnt + m_cnt + t_cnt} 辆\n\n")
        
        avg_conf_match = re.search(r'平均置信度.*?<div class="value">(.*?)<span', detection_stats, re.DOTALL)
        max_conf_match = re.search(r'最大置信度.*?<div class="value">(.*?)<span', detection_stats, re.DOTALL)
        
        if avg_conf_match:
            f.write(f"- 平均置信度: {avg_conf_match.group(1)}\n")
        if max_conf_match:
            f.write(f"- 最大置信度: {max_conf_match.group(1)}\n")
        
        f.write("\n")

        f.write("【语义分割结果】\n")
        f.write("-" * 50 + "\n")
        
        background_match = re.search(r'Background:\s*([^<]+)', segmentation_stats)
        water_match = re.search(r'Water:\s*([^<]+)', segmentation_stats)
        
        if background_match:
            f.write(f"- 背景像素比例: {background_match.group(1).strip()}\n")
        if water_match:
            f.write(f"- 水域像素比例: {water_match.group(1).strip()}\n")
    
    print(f"[保存] 联合分析结果已保存到: {file_path}")
    return file_path

def update_model_dropdown(category, subcategory):
    """更新模型选择下拉框"""
    models = get_available_models(category, subcategory)
    
    if isinstance(models[0], str) and ("无" in models[0] or "不存在" in models[0]):
        choices = models
        value = models[0]
        return gr.Dropdown(choices=choices, value=value, label="选择具体模型"), models, ""
    else:
        choices = [model["display"] for model in models]  
        value = choices[0] if choices else "无模型"

        size_html = ""
        if models and len(models) > 0:
            first_model = models[0]
            size_html = f"""
            <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                <span style="color:#666; font-size:13px;">模型大小：</span>
                {get_size_badge_html(first_model['size_mb'])}
            </div>
            """
        
        return gr.Dropdown(choices=choices, value=value, label="选择具体模型"), models, size_html

# 获取当前选择的模型信息
def get_current_model_info(category, subcategory, model_display, models):
    if isinstance(models, list) and models and isinstance(models[0], str) and ("无" in models[0] or "不存在" in models[0]):
        return models[0]
    if not models or not isinstance(models, list):
        return "无模型信息"
    
    for model in models:
        if model["display"] == model_display: 
            return model
    
    return "模型未找到"

# AI智能分析函数
def ai_analysis(api_key, message, chat_history, image):
    """AI智能分析函数"""
    if not api_key:
        return chat_history + [[message, "请输入有效的API密钥"]], ""
    ai_response = f"已分析您的查询: {message}"
    if image is not None:
        ai_response += "\n已处理上传的图像数据"
    
    ai_response += "\n\n分析结果: 这是一个示例响应，请配置真实的AI API接口。"
    
    return chat_history + [[message, ai_response]], ""

# 初始化数据库
init_db()
print("[启动] 数据库 YOLO_flooded 初始化完成")

# 启动时检查模型
try:
    model_paths = glob.glob(os.path.join("detection_models", "*.pt"))
    print(f"[启动] 检测模型文件: {[os.path.basename(p) for p in model_paths]}")
    if model_paths:
        test_model = YOLO(model_paths[0])
        names_count = len(test_model.names) if hasattr(test_model, 'names') else 0
        print(f"[启动] 模型验证: {os.path.basename(model_paths[0])} names_count={names_count}")
except Exception as ve:
    print(f"[启动] 模型验证跳过: {ve}")

# 创建Gradio界面
with gr.Blocks(
    title="水淹位置检测与水域分割可视化系统",
    theme=gr.themes.Soft(),
    css="""
    :root{
      --title-bg:#3F6BC8; --title-color:#fff;
      --card-bg:#ffffff; --card-radius:12px; --card-shadow:0 8px 22px rgba(16,24,40,.08);
    }
    #stats{display:grid; grid-template-columns:3fr 1fr 1fr; gap:16px; align-items:start;}
    .card{background:var(--card-bg); border-radius:var(--card-radius); box-shadow:var(--card-shadow); padding:16px;}
    .title{background:var(--title-bg); color:var(--title-color); padding:10px 14px; border-radius:8px; font-weight:700; margin-bottom:14px;}
    .badges{display:flex; gap:12px; flex-wrap:wrap;}
    .badge{padding:8px 14px; border-radius:8px; color:#000; font-weight:600; box-shadow: inset 0 0 0 1px rgba(0,0,0,.06);}
    .badge.bottom{background:#4CAF50;}
    .badge.middle{background:#FFE082;}
    .badge.top{background:#F44336;}
    .badge.water{background:#2196F3; color:#fff;} 
    .badge.background{background:#9E9E9E;}  
    .metric .value{font-size:28px; font-weight:800; color:#1f2937;}
    .metric .unit{margin-left:2px; font-size:14px; opacity:.7;}
    .badges3{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;   /* 三等分 */
      align-items:center;
      width:100%;
    }
    .badges3 .left   { justify-self:start;  }
    .badges3 .center { justify-self:center; }
    .badges3 .right  { justify-self:end;    }

    /* 新增：两栏布局用于语义分割 */
    .badges2{
      display:grid;
      grid-template-columns: 1fr 1fr;   /* 两等分 */
      gap: 12px;
      align-items:center;
      width:100%;
    }
    .badges2 .left   { justify-self:start;  }
    .badges2 .right  { justify-self:end;    }

/* 模态对话框样式 */
.yaml-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  padding: 20px;
}

.yaml-modal-content {
  background: white;
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  width: 90%;
  max-width: 1000px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.yaml-modal-header {
  background: var(--title-bg);
  color: var(--title-color);
  padding: 16px 24px;
  font-size: 18px;
  font-weight: 700;
  border-radius: 12px 12px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.yaml-modal-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}

.yaml-modal-footer {
  padding: 16px 24px;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: flex-end;
}

/* 移除Gradio默认边距 */
.yaml-modal-content > .block {
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
}

.yaml-code-content {
  margin: 0 !important;
  border-radius: 0 !important;
}
/* 覆盖/修正：全屏遮罩 */
.yaml-modal-overlay{
  position: fixed; inset: 0;
  background: rgba(0,0,0,.6);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999; padding: 20px;
}

/* 弹窗主体：只保留一层滚动在代码区 */
.yaml-modal-content{
  background:#fff; border-radius:12px; box-shadow:0 20px 60px rgba(0,0,0,.3);
  width: min(1000px, 90vw);
  max-height: 85vh;
  display:flex; flex-direction:column;
  overflow: hidden;          /* 关键：外层不滚动 */
}

/* 头部条 */
.yaml-modal-header{
  background: var(--title-bg); color: var(--title-color);
  padding: 14px 18px; font-weight: 700; border-radius: 12px 12px 0 0;
}

/* 代码滚动容器（唯一滚动条） */
.yaml-scroll{
  height: calc(85vh - 120px);
  overflow: auto;
  padding: 0; 
  margin: 0;
  background: #f9fafb;
}

/* 移除 Gradio 在弹窗里的默认外边距/内边距 */
.yaml-modal-content .block,
.yaml-modal-content .form,
.yaml-modal-content .row,
.yaml-modal-content .column{
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
}

/* 强制移除 Code 组件的内部滚动 */
.yaml-code,
.yaml-code > div,
.yaml-code .wrap,
.yaml-code .code-wrap{
  height: 100% !important;
  max-height: none !important;
  overflow: visible !important;
  border: none !important;
}

/* 针对 Code 组件内的 textarea 和 pre */
.yaml-scroll textarea,
.yaml-scroll pre,
.yaml-scroll code,
.yaml-scroll .cm-editor,
.yaml-scroll .cm-scroller{
  height: auto !important;
  max-height: none !important;
  overflow: visible !important;
  overflow-y: visible !important;
  overflow-x: visible !important;
}

/* 移除 CodeMirror 的滚动（如果使用了） */
.yaml-scroll .CodeMirror,
.yaml-scroll .CodeMirror-scroll{
  height: auto !important;
  overflow: visible !important;
}

/* 打开弹窗时禁止页面本体滚动（仅 CSS，无 JS） */
.gradio-container:has(.yaml-modal-overlay[style*="display: block"]){
  overflow: hidden !important;
}

/* 图片容器统一样式 */
.image-container {
    width: 100%;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.image-container .image {
    width: 100%;
    max-width: 400px;
    height: 300px;
    object-fit: contain;
}

/* 按钮水平对齐 */
.button-row {
    display: flex;
    gap: 12px;
    align-items: center;
    margin: 16px 0;
}

/* 加载状态按钮样式 */
.loading-btn {
    background: #3F6BC8 !important;
    color: white !important;
}

.loading-btn.success {
    background: #28a745 !important;
    color: white !important;
}

.loading-btn.error {
    background: #dc3545 !important;
    color: white !important;
}

/* 右侧内容区域 */
.right-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.model-info-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.history-section {
    background: #fff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 16px;
    max-height: 200px;
    overflow-y: auto;
}

/* 新增：车辆统计图片样式 */
.vehicle-stats-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 16px 0;
}

.vehicle-stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
}

.vehicle-image {
    width: 100%;
    max-width: 180px;
    height: 120px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid #e5e7eb;
    margin-bottom: 8px;
}

.vehicle-label {
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    text-align: center;
    min-width: 80px;
}

.vehicle-label.bottom {
    background: #4CAF50;
    color: white;
}

.vehicle-label.middle {
    background: #FFE082;
    color: #333;
}

.vehicle-label.top {
    background: #F44336;
    color: white;
}
"""
) as demo:
    gr.Markdown("""
    # 🎯 水淹位置检测与水域分割可视化系统
    
    本系统支持目标检测和语义分割两大类别，每个类别下包含多种模型架构。
    """)
    
    create_model_folders()
    
    # 状态存储
    current_detection_models = gr.State([])  
    current_segmentation_models = gr.State([])  
    loaded_detection_model = gr.State(None)  
    loaded_segmentation_model = gr.State(None)  
    

    combined_detection_models_state = gr.State([])
    combined_segmentation_models_state = gr.State([])
    loaded_combined_detection_model = gr.State(None)
    loaded_combined_segmentation_model = gr.State(None)
    
    yaml_fullscreen = gr.Group(visible=False, elem_classes="yaml-modal-overlay")
    with yaml_fullscreen:
        with gr.Column(elem_classes="yaml-modal-content", scale=1):
            gr.HTML("""<div class="yaml-modal-header">🧾 模型 YAML 配置</div>""")
            with gr.Group(elem_classes="yaml-scroll"):
                yaml_full_code = gr.Code(
                    language="yaml",
                    label=None,
                    lines=65,           
                    max_lines=999,      
                    interactive=False,
                    elem_classes="yaml-code" 
                )

            with gr.Row():  
                close_yaml_full_btn = gr.Button("✕ 关闭", variant="secondary", size="lg")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("🎯 目标检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置")
                    
                    with gr.Row():
                        detection_category = gr.Radio(
                            choices=["YOLO模型", "其他模型"],
                            value="YOLO模型",
                            label="检测模型类型"
                        )
                    
                    detection_model_dropdown = gr.Dropdown(
                        choices=["无模型"],
                        value="无模型",
                        label="选择具体模型",
                        allow_custom_value=False
                    )
                    
                    detection_model_size = gr.HTML(value="", label=None)

                    load_detection_btn = gr.Button("加载检测模型", variant="primary")
                    load_detection_status = gr.Textbox(
                        label="加载状态", 
                        interactive=False, 
                        lines=5, 
                        max_lines=10
                    )

                    view_yaml_btn = gr.Button("查看模型 YAML", variant="secondary")

                    gr.Markdown("#### YOLO 参数设置")
                    detection_conf_slider = gr.Slider(
                        minimum=0.001,
                        maximum=1.0,
                        value=0.01,
                        step=0.01,
                        label="置信度阈值 (conf)",
                        info="检测目标的最小置信度"
                    )
                    detection_iou_slider = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.45,
                        step=0.05,
                        label="IoU阈值 (iou)",
                        info="非极大值抑制的IoU阈值"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 图像处理")
                    
                    with gr.Tabs():
                        with gr.TabItem("📷 图像检测"):
                            with gr.Row():
                                with gr.Column(elem_classes="image-container"):
                                    detection_input = gr.Image(
                                        type="numpy", 
                                        label="上传检测图像", 
                                        height=300,
                                        width=400,
                                        elem_classes="image"
                                    )
                                    detection_btn = gr.Button("执行目标检测", variant="primary")
                                
                                with gr.Column(elem_classes="image-container"):
                                    detection_output = gr.Image(
                                        label="检测结果", 
                                        interactive=False, 
                                        height=300,
                                        width=400,
                                        elem_classes="image"
                                    )
                                    with gr.Group():
                                        detection_download_btn = gr.Button("📥 下载检测结果", variant="secondary")
                                        detection_download_file = gr.File(label="检测结果文件", visible=False)
                            
                            with gr.Group(elem_classes="model-info-card"):
                                gr.Markdown("#### 📊 模型信息")
                                model_info_display = gr.HTML(value="<p>请先加载模型查看详细信息</p>")
                            
                            img1_b64 = get_image_base64("assets/bottom.png")
                            img2_b64 = get_image_base64("assets/middle.png")
                            img3_b64 = get_image_base64("assets/top.png")

                            stats_panel = gr.HTML(value=f"""
<div id="stats">
  <div class="card">
    <div class="title">车辆淹没位置数量统计</div>
    <div class="vehicle-stats-container">
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img1_b64}" alt="Bottom" class="vehicle-image" />
        <div class="vehicle-label bottom">Bottom:(0)</div>
      </div>
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img2_b64}" alt="Middle" class="vehicle-image" />
        <div class="vehicle-label middle">middle:(0)</div>
      </div>
      <div class="vehicle-stat-item">
        <img src="data:image/png;base64,{img3_b64}" alt="Top" class="vehicle-image" />
        <div class="vehicle-label top">top:(0)</div>
      </div>
    </div>
  </div>
  <div class="card metric">
    <div class="title">平均置信度</div>
    <div class="value">-<span class="unit"></span></div>
  </div>
  <div class="card metric">
    <div class="title">最大置信度</div>
    <div class="value">-<span class="unit"></span></div>
  </div>
</div>
""")
                        
                        with gr.TabItem("🎬 视频检测"):
                            with gr.Row():
                                with gr.Column():
                                    video_detection_input = gr.Video(
                                        label="上传检测视频",
                                        height=400
                                    )
                                    video_detection_btn = gr.Button("执行视频检测", variant="primary", size="lg")
                                
                                with gr.Column():
                                    video_detection_output = gr.Video(
                                        label="检测结果视频（点击下载）",
                                        height=400,
                                        autoplay = False
                                    )
                                    with gr.Group():
                                        video_detection_download_btn = gr.Button("📥 下载视频检测结果统计", variant="secondary")
                                        video_detection_download_file = gr.File(label="视频检测结果文件", visible=False)
                            
                            video_stats_panel = gr.HTML(value=f"""
<div id="stats">
  <div class="card">
    <div class="title">视频检测统计</div>
    <p style="text-align:center; color:#666; padding:20px;">请上传视频并执行检测</p>
  </div>
</div>
""")

        with gr.TabItem("🔬 联合分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置")
                    gr.Markdown("#### 1️⃣ 目标检测模型")
                    combined_detection_category = gr.Radio(
                        choices=["YOLO模型", "其他模型"],
                        value="YOLO模型",
                        label="检测模型类型"
                    )
                    combined_detection_model = gr.Dropdown(
                        choices=["无模型"],
                        value="无模型",
                        label="选择检测模型"
                    )
                    combined_detection_size = gr.HTML(value="", label=None)
                    load_combined_detection_btn = gr.Button("加载检测模型", variant="primary", size="sm")
                    combined_detection_status = gr.Textbox(label="检测模型状态", interactive=False, lines=3)
                    
                    gr.Markdown("##### 检测参数")
                    combined_det_conf = gr.Slider(0, 1, 0.01, 0.01, label="检测置信度")
                    combined_det_iou = gr.Slider(0, 1, 0.45, 0.05, label="检测IoU")
                    
                    gr.Markdown("---")

                    gr.Markdown("#### 2️⃣ 语义分割模型")
                    combined_segmentation_category = gr.Radio(
                        choices=["YOLO模型", "其他模型"],
                        value="YOLO模型",
                        label="分割模型类型"
                    )
                    combined_segmentation_model = gr.Dropdown(
                        choices=["无模型"],
                        value="无模型",
                        label="选择分割模型"
                    )
                    combined_segmentation_size = gr.HTML(value="", label=None)
                    load_combined_segmentation_btn = gr.Button("加载分割模型", variant="primary", size="sm")
                    combined_segmentation_status = gr.Textbox(label="分割模型状态", interactive=False, lines=3)
                    
                    gr.Markdown("##### 分割参数")
                    combined_seg_conf = gr.Slider(0, 1, 0.15, 0.05, label="分割置信度")
                    combined_seg_iou = gr.Slider(0, 1, 0.45, 0.05, label="分割IoU")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 联合分析处理")              
                    with gr.Tabs():
                        with gr.TabItem("📷 图像联合分析"):
                            gr.Markdown("**处理流程**: 原始图像分别执行目标检测和语义分割")

                            gr.Markdown("#### 🖼️ 图像对比")
                            with gr.Row():
                                with gr.Column():
                                    combined_input = gr.Image(type="pil", label="原始图像", height=280)
                                
                                with gr.Column():
                                    combined_detection_result = gr.Image(label="目标检测结果", height=280)
                                
                                with gr.Column():
                                    combined_segmentation_result = gr.Image(label="语义分割结果", height=280)
                            
                            with gr.Row():
                                combined_run_btn = gr.Button("🚀 执行联合分析", variant="primary", size="lg")
                                combined_download_btn = gr.Button("📥 下载分析结果", variant="secondary")
                                combined_download_file = gr.File(label="分析结果文件", visible=False)

                            gr.Markdown("### 📊 目标检测统计")
                            combined_detection_stats = gr.HTML(value="""
<div style="padding: 16px; background: #f8f9fa; border-radius: 8px;">
    <p style="text-align:center; color:#666;">等待检测结果...</p>
</div>
""")

                            gr.Markdown("### 🎨 语义分割统计")
                            combined_segmentation_stats = gr.HTML(value="""
<div style="padding: 16px; background: #f8f9fa; border-radius: 8px;">
    <p style="text-align:center; color:#666;">等待分割结果...</p>
</div>
""")

                        with gr.TabItem("🎬 视频联合分析"):
                            gr.Markdown("**处理流程**: 对视频分别执行目标检测和语义分割（支持所有模型类型）")
                            
                            with gr.Row():
                                with gr.Column():
                                    combined_video_input = gr.Video(
                                        label="上传视频",
                                        height=350
                                    )
                                    combined_video_run_btn = gr.Button("🚀 执行视频联合分析", variant="primary", size="lg")
                                
                                with gr.Column():
                                    combined_video_detection_result = gr.Video(
                                        label="目标检测结果视频（点击下载）",
                                        height=350,
                                        autoplay=False
                                    )
                                
                                with gr.Column():
                                    combined_video_segmentation_result = gr.Video(
                                        label="语义分割结果视频（点击下载）",
                                        height=350,
                                        autoplay=False
                                    )

                            gr.Markdown("### 📊 目标检测统计")
                            combined_video_detection_stats = gr.HTML(value="""
<div style="padding: 16px; background: #f8f9fa; border-radius: 8px;">
    <p style="text-align:center; color:#666;">等待视频检测结果...</p>
</div>
""")

                            gr.Markdown("### 🎨 语义分割统计")
                            combined_video_segmentation_stats = gr.HTML(value="""
<div style="padding: 16px; background: #f8f9fa; border-radius: 8px;">
    <p style="text-align:center; color:#666;">等待视频分割结果...</p>
</div>
""")

        with gr.TabItem("🎨 语义分割"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置")
                    
                    with gr.Row():
                        segmentation_category = gr.Radio(
                            choices=["YOLO模型", "其他模型"],
                            value="YOLO模型", 
                            label="分割模型类型"
                        )
                    
                    segmentation_model_dropdown = gr.Dropdown(
                        choices=["无模型"],
                        value="无模型",
                        label="选择具体模型",
                        allow_custom_value=False
                    )
                    
                    segmentation_model_size = gr.HTML(value="", label=None)

                    load_segmentation_btn = gr.Button("加载分割模型", variant="primary")
                    load_segmentation_status = gr.Textbox(label="加载状态", interactive=False, lines=5, max_lines=10)
                
                    gr.Markdown("#### YOLO 参数设置")
                    segmentation_conf_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.15,
                        step=0.05,
                        label="置信度阈值 (conf)",
                        info="分割目标的最小置信度"
                    )
                    segmentation_iou_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.45,
                        step=0.05,
                        label="IoU阈值 (iou)",
                        info="非极大值抑制的IoU阈值"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 图像处理")
                    
                    with gr.Tabs():
                        with gr.TabItem("📷 图像分割"):
                            with gr.Row():
                                with gr.Column():
                                    segmentation_input = gr.Image(type="pil", label="上传分割图像", height=300)
                                    segmentation_btn = gr.Button("执行语义分割", variant="primary")
                                
                                with gr.Column():
                                    segmentation_output = gr.Image(label="分割结果", interactive=False, height=300)
                                    with gr.Group():
                                        segmentation_download_btn = gr.Button("📥 下载分割结果", variant="secondary")
                                        segmentation_download_file = gr.File(label="分割结果文件", visible=False)
                            
                            with gr.Group(elem_classes="model-info-card"):
                                gr.Markdown("#### 📊 模型信息")
                                segmentation_model_info_display = gr.HTML(value="<p>请先加载模型查看详细信息</p>")
                            
                            segmentation_stats_panel = gr.HTML(value="""
<div id="stats">
  <div class="card">
    <div class="title">语义分割像素占比统计</div>
    <div class="badges2">
      <div class="left">
        <span class="badge background">Background: -</span>
      </div>
      <div class="right">
        <span class="badge water">Water: -</span>
      </div>
    </div>
  </div>
  <div class="card metric">
    <div class="title">背景像素比例</div>
    <div class="value">-<span class="unit"></span></div>
  </div>
  <div class="card metric">
    <div class="title">水域像素比例</div>
    <div class="value">-<span class="unit"></span></div>
  </div>
</div>
""")

                        with gr.TabItem("🎬 视频分割"):
                            with gr.Row():
                                with gr.Column():
                                    video_segmentation_input = gr.Video(
                                        label="上传分割视频",
                                        height=400
                                    )
                                    video_segmentation_btn = gr.Button("执行视频分割", variant="primary", size="lg")
                                
                                with gr.Column():
                                    video_segmentation_output = gr.Video(
                                        label="分割结果视频（点击下载）",
                                        height=400,
                                        autoplay=False
                                    )

                            video_segmentation_stats_panel = gr.HTML(value="""
<div id="stats">
  <div class="card">
    <div class="title">视频分割统计</div>
    <p style="text-align:center; color:#666; padding:20px;">请上传视频并执行分割</p>
  </div>
</div>
""")
    
    # ===== 📊 历史记录 Tab =====
    with gr.TabItem("📊 历史记录"):
        with gr.Row():
            history_refresh_btn = gr.Button("🔄 刷新记录", variant="primary", size="lg")
        with gr.Row():
            detection_history_html = gr.HTML(value="""
<div id="stats">
  <div class="card">
    <div class="title">检测历史记录 (YOLO_flooded数据库)</div>
    <p style="text-align:center; color:#666; padding:20px;">点击刷新加载最近记录</p>
  </div>
</div>
""")
    
    # AI智能分析部分
    with gr.Accordion("🤖 AI智能分析助手", open=False):
        gr.Markdown("使用AI助手进行进一步的分析和对话")
        
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="API密钥",
                    type="password",
                    placeholder="请输入您的API密钥"
                )
                
                chat_image_input = gr.Image(type="pil", label="上传分析图片")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="AI对话", height=300, type="messages")
                chat_msg = gr.Textbox(label="输入您的问题", placeholder="请输入您要分析的问题...")
                
                with gr.Row():
                    chat_clear_btn = gr.Button("清除对话")
                    chat_submit_btn = gr.Button("发送消息", variant="primary")

    def update_detection_models(subcategory):
        return update_model_dropdown("目标检测", subcategory)
    
    detection_category.change(
        fn=update_detection_models,
        inputs=detection_category,
        outputs=[detection_model_dropdown, current_detection_models, detection_model_size]
        )

    def run_video_detection(video, subcategory, model_display, models, loaded_model, conf, iou):
        """执行视频检测"""
        if video is None:
            return None, "<p style='color:red;'>请先上传视频</p>", None, gr.update(visible=False)
            
        if loaded_model is None:
            return None, "<p style='color:red;'>请先加载模型</p>", None, gr.update(visible=False)
        
        try:
            if subcategory == "YOLO模型":
                output_video, stats, results_file = process_yolo_detection_video(
                    video, 
                    loaded_model, 
                    conf=conf, 
                    iou=iou
                )
            else:
                # 其他模型
                output_video, stats, results_file = process_other_detection_video(
                    video,
                    loaded_model,
                    conf=conf,
                    iou=iou
                )
            
            return output_video, stats, results_file, gr.update(visible=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"<p style='color:red;'>视频检测失败: {str(e)}</p>", None, gr.update(visible=False)

    def prepare_download_file(file_path):
        if file_path:
            return gr.update(value=file_path, visible=True)
        else:
            return gr.update(visible=False)

    def prepare_download_file(file_path):
        if file_path:
            return gr.update(value=file_path, visible=True)
        else:
            return gr.update(visible=False)
    
    video_detection_result_file_path = gr.State(None)
    
    video_detection_btn.click(
        fn=run_video_detection,
        inputs=[
                video_detection_input,
                detection_category,
                detection_model_dropdown,
                current_detection_models,
                loaded_detection_model,
                detection_conf_slider,
                detection_iou_slider
            ],
            outputs=[video_detection_output, video_stats_panel, video_detection_result_file_path, video_detection_download_file]
    )

    video_detection_download_btn.click(
        fn=prepare_download_file,
        inputs=[video_detection_result_file_path],
        outputs=[video_detection_download_file]
    )

    def update_detection_size_display(model_display, models):
        """当下拉框选择改变时更新模型大小显示"""
        if isinstance(models, list) and models and not isinstance(models[0], str):
            for model in models:
                if model["display"] == model_display:  
                    return f"""
                    <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                        <span style="color:#666; font-size:13px;">模型大小：</span>
                        {get_size_badge_html(model['size_mb'])}
                    </div>
                    """
        return ""

    detection_model_dropdown.change(
        fn=update_detection_size_display,
        inputs=[detection_model_dropdown, current_detection_models],
        outputs=detection_model_size
    )
    
    def load_detection_model_with_status(subcategory, model_display, models):
        model_info = get_current_model_info("目标检测", subcategory, model_display, models)
        status, model = load_model(model_info, subcategory, task_type="detection")

        if "成功" in status:
            button_class = "模型加载成功"
            status_color = "success"
        elif "失败" in status:
            button_class = "模型加载失败" 
            status_color = "error"
        else:
            button_class = "加载模型"
            status_color = "loading"
        
        if isinstance(model_info, dict) and "成功" in status:
            model_info_html = f"""
            <div style="padding: 12px; background: #e8f5e8; border-radius: 6px;">
                <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 模型已加载</h4>
                <p style="margin: 4px 0;"><strong>模型名称:</strong> {model_info.get('display', 'Unknown')}</p>
                <p style="margin: 4px 0;"><strong>文件大小:</strong> {model_info.get('size_mb', 0)}MB</p>
                <p style="margin: 4px 0;"><strong>类型:</strong> {subcategory}</p>
            </div>
            """
        else:
            model_info_html = f"""
            <div style="padding: 12px; background: #fff3cd; border-radius: 6px;">
                <h4 style="margin: 0 0 8px 0; color: #856404;">⚠️ 模型未加载</h4>
                <p style="margin: 4px 0;">请先选择并加载模型</p>
            </div>
            """
        
        return status, model, button_class, model_info_html

    load_detection_btn.click(
        fn=load_detection_model_with_status,
        inputs=[detection_category, detection_model_dropdown, current_detection_models],
        outputs=[load_detection_status, loaded_detection_model, load_detection_btn, model_info_display]
    )

    def preview_detection_yaml_text(subcategory, model_display, models):
        info = get_current_model_info("目标检测", subcategory, model_display, models)
        if isinstance(info, dict) and os.path.exists(info.get("yaml_filepath", "")):
            try:
                with open(info["yaml_filepath"], "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"# 读取失败: {e}"
        return "# 未找到对应的 YAML 文件"

    def open_yaml_fullscreen(subcategory, model_display, models):
        yaml_text = preview_detection_yaml_text(subcategory, model_display, models)
        return gr.update(selected="🎯 目标检测"), gr.update(visible=True), gr.update(value=yaml_text)

    def close_yaml_fullscreen():
        return gr.update(selected="🎯 目标检测"), gr.update(visible=False)


    view_yaml_btn.click(
        fn=open_yaml_fullscreen,
        inputs=[detection_category, detection_model_dropdown, current_detection_models],
        outputs=[main_tabs, yaml_fullscreen, yaml_full_code]
    )

    close_yaml_full_btn.click(
        fn=close_yaml_fullscreen,
        inputs=None,
        outputs=[main_tabs, yaml_fullscreen]
    )
    
    def run_detection(image, subcategory, model_display, models, loaded_model, conf, iou):
        model_info = get_current_model_info("目标检测", subcategory, model_display, models)
        
        if subcategory == "YOLO模型":
            result_image, stats_html = process_yolo_detection(image, model_info, loaded_model, conf, iou)
        else:
            result_image, stats_html = process_other_detection(image, model_info, loaded_model, conf, iou)

        download_path = None
        if result_image is not None:
            import re

            b_cnt_match = re.search(r'Bottom:\((\d+)\)', stats_html)
            m_cnt_match = re.search(r'middle:\((\d+)\)', stats_html)
            t_cnt_match = re.search(r'top:\((\d+)\)', stats_html)
            
            b_cnt = int(b_cnt_match.group(1)) if b_cnt_match else 0
            m_cnt = int(m_cnt_match.group(1)) if m_cnt_match else 0
            t_cnt = int(t_cnt_match.group(1)) if t_cnt_match else 0

            avg_conf_match = re.search(r'平均置信度.*?<div class="value">(.*?)<span', stats_html, re.DOTALL)
            max_conf_match = re.search(r'最大置信度.*?<div class="value">(.*?)<span', stats_html, re.DOTALL)
            
            avg_conf = avg_conf_match.group(1) if avg_conf_match else "-"
            max_conf = max_conf_match.group(1) if max_conf_match else "-"

            download_path = save_detection_results_to_file(b_cnt, m_cnt, t_cnt, avg_conf, max_conf)

        return result_image, stats_html, download_path, gr.update(visible=False)

    detection_result_file_path = gr.State(None)

    detection_btn.click(
        fn=run_detection,
        inputs=[
            detection_input, 
            detection_category, 
            detection_model_dropdown, 
            current_detection_models, 
            loaded_detection_model,
            detection_conf_slider,  
            detection_iou_slider    
        ],
        outputs=[detection_output, stats_panel, detection_result_file_path, detection_download_file]
    )

    detection_download_btn.click(
        fn=prepare_download_file,
        inputs=[detection_result_file_path],
        outputs=[detection_download_file]
    )

    def update_segmentation_models(subcategory):
        return update_model_dropdown("语义分割", subcategory)
    
    segmentation_category.change(
        fn=update_segmentation_models,
        inputs=segmentation_category,
        outputs=[segmentation_model_dropdown, current_segmentation_models, segmentation_model_size]
    )
    
    def update_segmentation_size_display(model_display, models):
        """当下拉框选择改变时更新模型大小显示"""
        if isinstance(models, list) and models and not isinstance(models[0], str):
            for model in models:
                if model["display"] == model_display:  
                    return f"""
                    <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                        <span style="color:#666; font-size:13px;">模型大小：</span>
                        {get_size_badge_html(model['size_mb'])}
                    </div>
                    """
        return ""

    segmentation_model_dropdown.change(
        fn=update_segmentation_size_display,
        inputs=[segmentation_model_dropdown, current_segmentation_models],
        outputs=segmentation_model_size
    )
    
    def load_segmentation_model_with_status(subcategory, model_display, models):
        model_info = get_current_model_info("语义分割", subcategory, model_display, models)
        status, model = load_model(model_info, subcategory, task_type="segmentation")
        
        if isinstance(model_info, dict) and "成功" in status:
            model_info_html = f"""
            <div style="padding: 12px; background: #e8f5e8; border-radius: 6px;">
                <h4 style="margin: 0 0 8px 0; color: #2d5a2d;">✅ 模型已加载</h4>
                <p style="margin: 4px 0;"><strong>模型名称:</strong> {model_info.get('display', 'Unknown')}</p>
                <p style="margin: 4px 0;"><strong>文件大小:</strong> {model_info.get('size_mb', 0)}MB</p>
                <p style="margin: 4px 0;"><strong>类型:</strong> {subcategory}</p>
            </div>
            """
        else:
            model_info_html = f"""
            <div style="padding: 12px; background: #fff3cd; border-radius: 6px;">
                <h4 style="margin: 0 0 8px 0; color: #856404;">⚠️ 模型未加载</h4>
                <p style="margin: 4px 0;">请先选择并加载模型</p>
            </div>
            """
        
        return status, model, model_info_html
    
    load_segmentation_btn.click(
        fn=load_segmentation_model_with_status,
        inputs=[segmentation_category, segmentation_model_dropdown, current_segmentation_models],
        outputs=[load_segmentation_status, loaded_segmentation_model, segmentation_model_info_display]
    )
    
    def run_segmentation(image, subcategory, model_display, models, loaded_model, conf, iou):
        model_info = get_current_model_info("语义分割", subcategory, model_display, models)
 
        if subcategory == "YOLO模型":
            result_image, stats_html = process_yolo_segmentation(image, model_info, loaded_model, conf, iou)
        else:
            result_image, stats_html = process_other_segmentation(image, model_info, loaded_model)

        download_path = None
        if result_image is not None:
            import re
            background_match = re.search(r'Background:\s*([^<]+)', stats_html)
            water_match = re.search(r'Water:\s*([^<]+)', stats_html)
            
            background_ratio = background_match.group(1).strip() if background_match else "-"
            water_ratio = water_match.group(1).strip() if water_match else "-"

            download_path = save_segmentation_results_to_file(background_ratio, water_ratio)

        return result_image, stats_html, download_path, gr.update(visible=False)

    segmentation_result_file_path = gr.State(None)
    
    segmentation_btn.click(
        fn=run_segmentation,
        inputs=[
            segmentation_input, 
            segmentation_category, 
            segmentation_model_dropdown, 
            current_segmentation_models, 
            loaded_segmentation_model,
            segmentation_conf_slider,  
            segmentation_iou_slider    
        ],
        outputs=[segmentation_output, segmentation_stats_panel, segmentation_result_file_path, segmentation_download_file]
    )

    segmentation_download_btn.click(
        fn=prepare_download_file,
        inputs=[segmentation_result_file_path],
        outputs=[segmentation_download_file]
    )

    def run_video_segmentation(video, subcategory, model_display, models, loaded_model, conf, iou, progress=gr.Progress()):
        """执行视频分割"""
        if video is None:
            return None, "<p style='color:red;'>请先上传视频</p>"
            
        if loaded_model is None:
            return None, "<p style='color:red;'>请先加载模型</p>"
        
        try:
            if subcategory == "YOLO模型":
                output_video, stats = process_yolo_segmentation_video(
                    video, 
                    loaded_model, 
                    conf=conf, 
                    iou=iou,
                    progress=progress  
                )
            else:
                # 其他模型
                output_video, stats = process_other_segmentation_video(
                    video,
                    loaded_model,
                    conf=conf,
                    iou=iou,
                    progress=progress
                )
            
            return output_video, stats
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"<p style='color:red;'>视频分割失败: {str(e)}</p>"
    
    video_segmentation_btn.click(
        fn=run_video_segmentation,
        inputs=[
            video_segmentation_input,
            segmentation_category,
            segmentation_model_dropdown,
            current_segmentation_models,
            loaded_segmentation_model,
            segmentation_conf_slider,
            segmentation_iou_slider
        ],
        outputs=[video_segmentation_output, video_segmentation_stats_panel]
    )

    def load_detection_history():
        """从数据库加载检测历史记录"""
        try:
            records = get_all_detection_results(limit=20)
            if not records:
                return """
<div id="stats">
  <div class="card">
    <div class="title">检测历史记录 (YOLO_flooded数据库)</div>
    <p style="text-align:center; color:#999; padding:20px;">暂无记录</p>
  </div>
</div>"""
            html_rows = []
            for r in records:
                html_rows.append(f"""
<tr>
  <td>{r['id']}</td>
  <td>{r['image_name']}</td>
  <td style="color:green;font-weight:bold;">{r['bottom_count']}</td>
  <td style="color:orange;font-weight:bold;">{r['middle_count']}</td>
  <td style="color:red;font-weight:bold;">{r['top_count']}</td>
  <td>{r['total_count']}</td>
  <td>{r['avg_conf']}</td>
  <td>{r['max_conf']}</td>
  <td style="font-size:12px;">{r['created_at']}</td>
</tr>""")
            table = f"""
<div id="stats">
  <div class="card">
    <div class="title">检测历史记录 (YOLO_flooded数据库)</div>
    <table style="width:100%;border-collapse:collapse;margin-top:12px;font-size:13px;">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="padding:8px;border:1px solid #ddd;">ID</th>
          <th style="padding:8px;border:1px solid #ddd;">图片名</th>
          <th style="padding:8px;border:1px solid #ddd;">Bottom</th>
          <th style="padding:8px;border:1px solid #ddd;">Middle</th>
          <th style="padding:8px;border:1px solid #ddd;">Top</th>
          <th style="padding:8px;border:1px solid #ddd;">总计</th>
          <th style="padding:8px;border:1px solid #ddd;">平均置信度</th>
          <th style="padding:8px;border:1px solid #ddd;">最大置信度</th>
          <th style="padding:8px;border:1px solid #ddd;">时间</th>
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
    <p style="text-align:right;font-size:12px;color:#999;margin-top:8px;">共 {len(records)} 条记录</p>
  </div>
</div>"""
            return table
        except Exception as e:
            return f"<p style='color:red;'>加载失败: {e}</p>"

    history_refresh_btn.click(
        fn=load_detection_history,
        inputs=None,
        outputs=detection_history_html
    )

    def update_combined_detection_models(subcategory):
        return update_model_dropdown("目标检测", subcategory)
    
    combined_detection_category.change(
        fn=update_combined_detection_models,
        inputs=combined_detection_category,
        outputs=[combined_detection_model, combined_detection_models_state, combined_detection_size]
    )
    

    def update_combined_segmentation_models(subcategory):
        return update_model_dropdown("语义分割", subcategory)
    
    combined_segmentation_category.change(
        fn=update_combined_segmentation_models,
        inputs=combined_segmentation_category,
        outputs=[combined_segmentation_model, combined_segmentation_models_state, combined_segmentation_size]
    )

    def update_combined_detection_size_display(model_display, models):
        if isinstance(models, list) and models and not isinstance(models[0], str):
            for model in models:
                if model["display"] == model_display:
                    return f"""
                    <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                        <span style="color:#666; font-size:13px;">模型大小：</span>
                        {get_size_badge_html(model['size_mb'])}
                    </div>
                    """
        return ""
    
    combined_detection_model.change(
        fn=update_combined_detection_size_display,
        inputs=[combined_detection_model, combined_detection_models_state],
        outputs=combined_detection_size
    )
    
    def update_combined_segmentation_size_display(model_display, models):
        if isinstance(models, list) and models and not isinstance(models[0], str):
            for model in models:
                if model["display"] == model_display:
                    return f"""
                    <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                        <span style="color:#666; font-size:13px;">模型大小：</span>
                        {get_size_badge_html(model['size_mb'])}
                    </div>
                    """
        return ""
    
    combined_segmentation_model.change(
        fn=update_combined_segmentation_size_display,
        inputs=[combined_segmentation_model, combined_segmentation_models_state],
        outputs=combined_segmentation_size
    )
    
    def load_combined_detection_with_status(subcategory, model_display, models):
        model_info = get_current_model_info("目标检测", subcategory, model_display, models)
        status, model = load_model(model_info, subcategory, task_type="detection")
        return status, model
    
    load_combined_detection_btn.click(
        fn=load_combined_detection_with_status,
        inputs=[combined_detection_category, combined_detection_model, combined_detection_models_state],
        outputs=[combined_detection_status, loaded_combined_detection_model]
    )
    
    def load_combined_segmentation_with_status(subcategory, model_display, models):
        model_info = get_current_model_info("语义分割", subcategory, model_display, models)
        status, model = load_model(model_info, subcategory, task_type="segmentation")
        return status, model
    
    load_combined_segmentation_btn.click(
        fn=load_combined_segmentation_with_status,
        inputs=[combined_segmentation_category, combined_segmentation_model, combined_segmentation_models_state],
        outputs=[combined_segmentation_status, loaded_combined_segmentation_model]
    )

    combined_result_file_path = gr.State(None)
    
    combined_run_btn.click(
        fn=process_combined_analysis,
        inputs=[
            combined_input,
            combined_detection_category,
            combined_detection_model,
            combined_detection_models_state,
            loaded_combined_detection_model,
            combined_det_conf,
            combined_det_iou,
            combined_segmentation_category,
            combined_segmentation_model,
            combined_segmentation_models_state,
            loaded_combined_segmentation_model,
            combined_seg_conf,
            combined_seg_iou
        ],
        outputs=[
            combined_detection_result,
            combined_segmentation_result,
            combined_detection_stats,
            combined_segmentation_stats,
            combined_result_file_path,
            combined_download_file
        ]
    )
    
    combined_download_btn.click(
        fn=prepare_download_file,
        inputs=[combined_result_file_path],
        outputs=[combined_download_file]
    )

    combined_video_run_btn.click(
        fn=process_combined_video_analysis,
        inputs=[
            combined_video_input,
            combined_detection_category,
            combined_detection_model,
            combined_detection_models_state,
            loaded_combined_detection_model,
            combined_det_conf,
            combined_det_iou,
            combined_segmentation_category,
            combined_segmentation_model,
            combined_segmentation_models_state,
            loaded_combined_segmentation_model,
            combined_seg_conf,
            combined_seg_iou
        ],
        outputs=[
            combined_video_detection_result,
            combined_video_segmentation_result,
            combined_video_detection_stats,
            combined_video_segmentation_stats
        ]
    )
    
    # AI对话处理
    def process_user_message(api_key, message, chat_history, image):
        if not message.strip():
            return chat_history, "", None
        
        user_message_content = message
        if image is not None:
            user_message_content += " [已上传图片]"

        chat_history.append({"role": "user", "content": user_message_content})

        ai_response = f"已收到您的消息: {message}"
        if image is not None:
            ai_response += " 和图片数据"
        ai_response += "。这是一个示例响应。"

        chat_history.append({"role": "assistant", "content": ai_response})
        
        return chat_history, "", None
    
    chat_msg.submit(
        fn=process_user_message,
        inputs=[api_key_input, chat_msg, chatbot, chat_image_input],
        outputs=[chatbot, chat_msg, chat_image_input]
    )
    
    chat_submit_btn.click(
        fn=process_user_message,
        inputs=[api_key_input, chat_msg, chatbot, chat_image_input],
        outputs=[chatbot, chat_msg, chat_image_input]
    )
    
    chat_clear_btn.click(lambda: [], None, chatbot, queue=False)

    def initialize_interface():
        detection_models = get_available_models("目标检测", "YOLO模型")
        detection_size_html = ""
        if isinstance(detection_models[0], str) and ("无" in detection_models[0] or "不存在" in detection_models[0]):
            detection_choices = detection_models
            detection_value = detection_models[0]
        else:
            detection_choices = [model["display"] for model in detection_models] 
            detection_value = detection_choices[0] if detection_choices else "无模型"
            if detection_models:
                detection_size_html = f"""
                <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                    <span style="color:#666; font-size:13px;">模型大小：</span>
                    {get_size_badge_html(detection_models[0]['size_mb'])}
                </div>
                """

        segmentation_models = get_available_models("语义分割", "YOLO模型")
        segmentation_size_html = ""
        if isinstance(segmentation_models[0], str) and ("无" in segmentation_models[0] or "不存在" in segmentation_models[0]):
            segmentation_choices = segmentation_models
            segmentation_value = segmentation_models[0]
        else:
            segmentation_choices = [model["display"] for model in segmentation_models]  
            segmentation_value = segmentation_choices[0] if segmentation_choices else "无模型"
            if segmentation_models:
                segmentation_size_html = f"""
                <div style="margin-top:8px; padding:8px; background:#f8f9fa; border-radius:6px;">
                    <span style="color:#666; font-size:13px;">模型大小：</span>
                    {get_size_badge_html(segmentation_models[0]['size_mb'])}
                </div>
                """
        
        return (
            gr.Dropdown(choices=detection_choices, value=detection_value, allow_custom_value=False),
            detection_models,
            detection_size_html,
            gr.Dropdown(choices=segmentation_choices, value=segmentation_value, allow_custom_value=False),
            segmentation_models,
            segmentation_size_html,
            gr.Dropdown(choices=detection_choices, value=detection_value, allow_custom_value=False),
            detection_models,
            detection_size_html,
            gr.Dropdown(choices=segmentation_choices, value=segmentation_value, allow_custom_value=False),
            segmentation_models,
            segmentation_size_html
        )
    
    demo.load(
        fn=initialize_interface,
        outputs=[
            # 目标检测标签页
            detection_model_dropdown, 
            current_detection_models,
            detection_model_size,
            # 语义分割标签页
            segmentation_model_dropdown,
            current_segmentation_models,
            segmentation_model_size,
            # 联合分析标签页
            combined_detection_model,
            combined_detection_models_state,
            combined_detection_size,
            combined_segmentation_model,
            combined_segmentation_models_state,
            combined_segmentation_size
        ]
    )

# 启动界面
if __name__ == "__main__":
    create_model_folders()
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            inbrowser=True
        )
    except Exception as e:
        print(f"启动失败: {e}")