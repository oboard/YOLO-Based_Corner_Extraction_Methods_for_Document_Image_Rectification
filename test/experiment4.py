# test IoU between pose-old-best.pt and pose-new-best.pt and segment-best.pt
# 1. datasets主要有三个目录：
# •  images/：包含所有图片数据
# •  pose-labels/：包含姿态标注数据（JSON文件）
# •  segment-labels/：包含分割标注数据（JSON文件）
# 2. images/ 目录下有5个子目录：
# •  complex/：复杂背景图片
# •  dark_pure/：深色纯色背景图片
# •  shadow_light/：带阴影的光照图片
# •  white_pure/：白色纯色背景图片
# •  wood_pure/：木纹纯色背景图片
# 3. 每个背景类型目录下都有6个子目录：
# •  35cm/：35厘米距离拍摄的图片
# •  45cm/：45厘米距离拍摄的图片
# •  55cm/：55厘米距离拍摄的图片
# •  右下角正对手持/：右下角正对手持拍摄的图片
# •  左下角正对手持/：左下角正对手持拍摄的图片
# •  底边正对手持/：底边正对手持拍摄的图片
# 4. 每个子目录下包含一系列按序号命名的jpg图片文件（如00001.jpg, 00002.jpg等）
# 5. pose-labels/ 和 segment-labels/ 目录包含与图片对应的JSON格式标注文件，文件名与图片序号对应
# experiment_results.csv 中是已经跑完的数据


# 现在需要到处一些结果图，展示预测校正后的效果，做成一组 Grid 图，每组图包含 3 张图，分别是：
# 1. 原始图片加四边形，带点和边
# 2. 通过透视变换校正后的图片

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于 macOS
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('experiment_results.csv')

# 从 DataFrame 中随机抽取 10 行
if len(df) > 10:
    df_sampled = df.sample(10, random_state=40) # 使用固定 random_state 以便重复实验
else:
    df_sampled = df

# --- 将IoU值添加到图片标题中 --- #
def add_iou_to_title(title, iou_value):
    if not pd.isna(iou_value):
        return f'{title}\nIoU: {iou_value:.3f}'
    return title

# 输出目录
output_dir = Path('output/visualizations/pose_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

image_dir = Path('datasets/images')
pose_label_dir = Path('datasets/pose-labels')

# 遍历抽样后的数据生成可视化图
for index, row in df_sampled.iterrows():
    # 确保文件名格式正确（带有前导零的五位数字）
    img_stem = str(row['image'])
    img_name_formatted = f'{int(img_stem):05d}'
    
    background = row['background']
    distance = row['distance']
    
    img_path = image_dir / background / distance / f'{img_name_formatted}.jpg'
    pose_label_path = pose_label_dir / f'{img_name_formatted}.json'
    
    if not img_path.exists() or not pose_label_path.exists():
        print(f"文件不存在，跳过: {img_path} 或 {pose_label_path}")
        continue
    
    # 读取原始图片
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换为RGB以便 matplotlib 显示
    img_height, img_width, _ = img.shape
    
    # --- 绘制真实标注 --- #
    img_gt = img.copy()
    true_points = {}
    try:
        with open(pose_label_path, 'r') as f:
            true_data = json.load(f)
        # pose 标注格式：使用四个角点
        for shape in true_data['shapes']:
            if shape['shape_type'] == 'point':
                true_points[shape['label']] = shape['points'][0]
                
        if all(corner in true_points for corner in ['T_L', 'T_R', 'B_R', 'B_L']):
             # 按顺序排列角点：左上、右上、右下、左下
            points = np.array([
                true_points['T_L'],
                true_points['T_R'],
                true_points['B_R'],
                true_points['B_L']
            ])
            # 将坐标缩放到原始图片尺寸
            points[:, 0] = points[:, 0] * img_width / true_data['imageWidth']
            points[:, 1] = points[:, 1] * img_height / true_data['imageHeight']
            points = points.astype(np.int32)
            
            # 绘制四边形，增加线条粗细
            cv2.polylines(img_gt, [points], True, (0, 255, 0), 20) # 绿色，粗细改为 4
            # 绘制点，增加点的大小
            for point in points:
                 cv2.circle(img_gt, tuple(point), 50, (255, 0, 0), -1) # 蓝色，半径改为 10
        else:
            print(f"警告：标注文件 {pose_label_path} 未找到所有四个角点")

    except Exception as e:
        print(f"读取或绘制真实标注时出错 {pose_label_path}: {str(e)}")
        img_gt = img.copy()
    
    # --- 绘制 Pose Origin 预测 --- #
    img_pose_origin_pred = img.copy()
    if not pd.isna(row['pose-old_pred_T_L_x']):
        pred_points_origin_640 = np.array([
            [row['pose-old_pred_T_L_x'], row['pose-old_pred_T_L_y']],
            [row['pose-old_pred_T_R_x'], row['pose-old_pred_T_R_y']],
            [row['pose-old_pred_B_R_x'], row['pose-old_pred_B_R_y']],
            [row['pose-old_pred_B_L_x'], row['pose-old_pred_B_L_y']]
        ], dtype=np.float32)

        # 将 640x640 的预测点缩放到原始图像尺寸
        pred_points_origin = pred_points_origin_640.copy()
        pred_points_origin[:, 0] = pred_points_origin[:, 0] * img_width / 640
        pred_points_origin[:, 1] = pred_points_origin[:, 1] * img_height / 640
        pred_points_origin = pred_points_origin.astype(np.int32)

        # 绘制四边形，增加线条粗细
        cv2.polylines(img_pose_origin_pred, [pred_points_origin], True, (255, 165, 0), 20) # 橙色，粗细改为 4
        # 绘制点，增加点的大小
        for point in pred_points_origin:
             cv2.circle(img_pose_origin_pred, tuple(point), 50, (0, 0, 255), -1) # 红色，半径改为 10
    else:
        print(f"警告：图片 {img_name_formatted} 没有 Pose Origin 预测点")
    
    # --- 绘制 Pose GSCL 预测 --- #
    img_pose_gscl_pred = img.copy()
    if not pd.isna(row['pose-new_pred_T_L_x']):
        pred_points_gscl_640 = np.array([
            [row['pose-new_pred_T_L_x'], row['pose-new_pred_T_L_y']],
            [row['pose-new_pred_T_R_x'], row['pose-new_pred_T_R_y']],
            [row['pose-new_pred_B_R_x'], row['pose-new_pred_B_R_y']],
            [row['pose-new_pred_B_L_x'], row['pose-new_pred_B_L_y']]
        ], dtype=np.float32)
        
        # 将 640x640 的预测点缩放到原始图像尺寸
        pred_points_gscl = pred_points_gscl_640.copy()
        pred_points_gscl[:, 0] = pred_points_gscl[:, 0] * img_width / 640
        pred_points_gscl[:, 1] = pred_points_gscl[:, 1] * img_height / 640
        pred_points_gscl = pred_points_gscl.astype(np.int32)

        # 绘制四边形，增加线条粗细
        cv2.polylines(img_pose_gscl_pred, [pred_points_gscl], True, (255, 165, 0), 20) # 橙色，粗细改为 4
        # 绘制点，增加点的大小
        for point in pred_points_gscl:
             cv2.circle(img_pose_gscl_pred, tuple(point), 50, (0, 0, 255), -1) # 红色，半径改为 10
    else:
        print(f"警告：图片 {img_name_formatted} 没有 Pose GSCL 预测点")

    # --- 创建并保存图表 --- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_gt)
    axes[0].set_title('Ground Truth', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img_pose_origin_pred)
    axes[1].set_title(add_iou_to_title('Pose Origin Prediction', row['pose-old_iou']), fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(img_pose_gscl_pred)
    axes[2].set_title(add_iou_to_title('Pose GSCL Prediction', row['pose-new_iou']), fontsize=10)
    axes[2].axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_dir / f'{img_name_formatted}_pose_comparison.png', dpi=300)
    plt.close(fig)

print(f"可视化图已生成并保存到 {output_dir}")