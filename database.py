"""
SQLite数据库模块 - 存储检测/分割结果到 YOLO_flooded 数据库
"""
import sqlite3
import os
import base64
from datetime import datetime

# 数据库文件路径（项目根目录）
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "YOLO_flooded.db")


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建表结构（如果不存在）"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 检测结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_data BLOB,
            bottom_count INTEGER DEFAULT 0,
            middle_count INTEGER DEFAULT 0,
            top_count INTEGER DEFAULT 0,
            total_count INTEGER DEFAULT 0,
            avg_conf TEXT DEFAULT '-',
            max_conf TEXT DEFAULT '-',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 分割结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS segmentation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_data BLOB,
            water_ratio REAL DEFAULT 0.0,
            background_ratio REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"[数据库] 初始化完成: {DB_PATH}")


def save_detection_result(image_name, image_bytes, b_cnt, m_cnt, t_cnt, avg_conf, max_conf):
    """保存检测结果到数据库"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_results 
            (image_name, image_data, bottom_count, middle_count, top_count, total_count, avg_conf, max_conf)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            image_bytes,
            b_cnt,
            m_cnt,
            t_cnt,
            b_cnt + m_cnt + t_cnt,
            avg_conf,
            max_conf
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        print(f"[数据库] 检测结果已保存 ID={record_id}, 图片={image_name}")
        return record_id
    except Exception as e:
        print(f"[数据库] 保存检测结果失败: {e}")
        return None


def save_segmentation_result(image_name, image_bytes, water_ratio, background_ratio):
    """保存分割结果到数据库"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO segmentation_results
            (image_name, image_data, water_ratio, background_ratio)
            VALUES (?, ?, ?, ?)
        """, (image_name, image_bytes, water_ratio, background_ratio))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        print(f"[数据库] 分割结果已保存 ID={record_id}, 图片={image_name}")
        return record_id
    except Exception as e:
        print(f"[数据库] 保存分割结果失败: {e}")
        return None


def get_all_detection_results(limit=50):
    """获取最近的检测结果列表"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, image_name, bottom_count, middle_count, top_count, total_count, 
                   avg_conf, max_conf, created_at
            FROM detection_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[数据库] 查询检测结果失败: {e}")
        return []


def get_all_segmentation_results(limit=50):
    """获取最近的分割结果列表"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, image_name, water_ratio, background_ratio, created_at
            FROM segmentation_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[数据库] 查询分割结果失败: {e}")
        return []