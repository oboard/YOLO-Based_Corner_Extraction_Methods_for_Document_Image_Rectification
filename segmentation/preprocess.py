import os
import json
import shutil
import random
from pathlib import Path

def convert_labelme_to_yolo_segment(json_file, img_width, img_height):
    """将labelme格式的JSON转换为YOLO分割格式的txt文件内容"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue
        
        # 获取标签ID (本例中只有一个类别 'paper')
        class_id = 0
        
        # 获取多边形坐标点
        points = shape['points']
        
        # 转换为YOLO格式 (归一化坐标)
        yolo_points = []
        for point in points:
            x, y = point
            # 归一化坐标 (0-1范围)
            x_norm = x / img_width
            y_norm = y / img_height
            yolo_points.extend([x_norm, y_norm])
        
        # 格式化为YOLO分割格式: class_id x1 y1 x2 y2 ...
        line = str(class_id) + ' ' + ' '.join([f"{p:.6f}" for p in yolo_points])
        results.append(line)
    
    return '\n'.join(results)

def preprocess_dataset(dataset_path, train_ratio=0.8, random_seed=42):
    """预处理数据集，创建YOLO分割所需的目录结构"""
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建目录结构
    base_dir = Path(dataset_path)
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    
    # 创建图像和标签目录
    train_images_dir = train_dir / 'images'
    train_labels_dir = train_dir / 'labels'
    val_images_dir = val_dir / 'images'
    val_labels_dir = val_dir / 'labels'
    
    # 确保目录存在
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(Path(base_dir / 'labels').glob('*.json'))
    
    # 随机打乱文件列表
    random.shuffle(json_files)
    
    # 划分训练集和验证集
    split_idx = int(len(json_files) * train_ratio)
    train_jsons = json_files[:split_idx]
    val_jsons = json_files[split_idx:]
    
    print(f"总文件数: {len(json_files)}")
    print(f"训练集文件数: {len(train_jsons)}")
    print(f"验证集文件数: {len(val_jsons)}")
    
    # 处理训练集
    process_files(train_jsons, base_dir, train_images_dir, train_labels_dir)
    
    # 处理验证集
    process_files(val_jsons, base_dir, val_images_dir, val_labels_dir)
    
    print("数据集预处理完成！")

def process_files(json_files, base_dir, images_dir, labels_dir):
    """处理文件并复制到目标目录"""
    for json_file in json_files:
        # 获取文件名（不包含扩展名）
        file_stem = json_file.stem
        
        # 查找对应的图像文件
        img_file = list(Path(base_dir / 'images').glob(f"{file_stem}.*"))
        if not img_file:
            print(f"警告: 未找到 {file_stem} 对应的图像文件")
            continue
        
        img_file = img_file[0]
        
        # 读取JSON文件中的图像尺寸
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            img_width = data.get('imageWidth', 0)
            img_height = data.get('imageHeight', 0)
        
        if img_width == 0 or img_height == 0:
            print(f"警告: {file_stem} 中未找到有效的图像尺寸信息")
            continue
        
        # 转换为YOLO格式
        yolo_content = convert_labelme_to_yolo_segment(json_file, img_width, img_height)
        
        # 创建YOLO格式的标签文件
        txt_file = labels_dir / f"{file_stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(yolo_content)
        
        # 复制图像文件
        shutil.copy2(img_file, images_dir / img_file.name)
        print(f"处理: {file_stem}")

if __name__ == "__main__":
    # 预处理数据集
    preprocess_dataset('datasets') 