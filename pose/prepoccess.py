import os
import shutil
from pathlib import Path

def is_valid_label_file(label_file):
    """检查标签文件是否有效"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if not lines:  # 空文件无效
                return False
            
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                values = [float(x) for x in line.split()]
                if len(values) != 11:  # 确保有11个值（1个类别 + 5个点的坐标）
                    return False
        return True
    except Exception:
        return False

def copy_images(src_dir, dst_dir, label_dir):
    """复制图片文件到目标目录，只复制有效标签对应的图片"""
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)
    
    # 遍历所有图片文件
    for ext in image_extensions:
        for img_file in Path(src_dir).glob(f'**/*{ext}'):
            try:
                # 获取对应的标签文件路径
                rel_path = img_file.relative_to(src_dir)
                label_file = Path(label_dir) / rel_path.with_suffix('.txt')
                
                # 检查标签文件是否存在且有效
                if not label_file.exists() or not is_valid_label_file(label_file):
                    print(f"跳过图片 {img_file}：标签文件无效或不存在")
                    continue
                
                # 创建目标路径
                dst_path = os.path.join(dst_dir, str(rel_path))
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # 复制文件
                shutil.copy2(img_file, dst_path)
            except Exception as e:
                print(f"复制图片文件 {img_file} 时出错: {str(e)}")

def convert_to_yolo_pose_format(input_dir, output_dir):
    """
    将数据集转换为YOLO Pose格式
    输入格式: class x1 y1 w1 h1 x2 y2 x3 y3 x4 y4 x5 y5
    输出格式: class x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有标签文件
    for label_file in Path(input_dir).glob('**/*.txt'):
        try:
            # 读取原始标签
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:  # 跳过空文件
                    continue
                
                # 处理每一行
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                        
                    values = [float(x) for x in line.split()]
                    if len(values) != 11:  # 确保有11个值（1个类别 + 5个点的坐标）
                        print(f"警告: 文件 {label_file} 格式不正确，跳过")
                        continue
                    
                    # 提取类别和坐标
                    class_id = int(values[0])
                    # 只取前4个点的坐标（跳过第5个点）
                    coords = values[1:9]
                    
                    # 转换为YOLO Pose格式
                    # 添加可见性标记（假设所有点都是可见的）
                    pose_values = [class_id]
                    for i in range(0, len(coords), 2):
                        pose_values.extend([coords[i], coords[i+1], 1.0])
                    
                    new_lines.append(' '.join(map(str, pose_values)))
                
                if not new_lines:  # 如果没有有效的行，跳过
                    continue
                
                # 创建输出文件路径
                rel_path = label_file.relative_to(input_dir)
                output_path = os.path.join(output_dir, str(rel_path))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 写入新格式的标签
                with open(output_path, 'w') as f:
                    f.write('\n'.join(new_lines))
                    
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {str(e)}")
            continue

def process_dataset():
    # 处理训练集
    train_input = 'datasets1/train/labels'
    train_output = 'datasets/train/labels'
    train_images_src = 'datasets1/train/images'
    train_images_dst = 'datasets/train/images'
    convert_to_yolo_pose_format(train_input, train_output)
    copy_images(train_images_src, train_images_dst, train_input)
    
    # 处理验证集
    valid_input = 'datasets1/valid/labels'
    valid_output = 'datasets/val/labels'
    valid_images_src = 'datasets1/valid/images'
    valid_images_dst = 'datasets/val/images'
    convert_to_yolo_pose_format(valid_input, valid_output)
    copy_images(valid_images_src, valid_images_dst, valid_input)
    
    # 处理测试集
    test_input = 'datasets1/test/labels'
    test_output = 'datasets/test/labels'
    test_images_src = 'datasets1/test/images'
    test_images_dst = 'datasets/test/images'
    convert_to_yolo_pose_format(test_input, test_output)
    copy_images(test_images_src, test_images_dst, test_input)
    
    print("数据集转换完成！")

if __name__ == '__main__':
    process_dataset()
