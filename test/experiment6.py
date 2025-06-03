import pandas as pd
import matplotlib.pyplot as plt
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
    df_sampled = df.sample(10, random_state=42)  # 使用固定 random_state 以便重复实验
else:
    df_sampled = df

# --- 将IoU值添加到图片标题中 --- #
def add_iou_to_title(title, iou_value):
    if not pd.isna(iou_value):
        return f'{title}\nIoU: {iou_value:.3f}'
    return title

# 执行透视变换
def perform_perspective_transform(img, src_points):
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 计算矫正后图像的大小
    width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                       ((src_points[1][1] - src_points[0][1]) ** 2))
    width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                          ((src_points[2][1] - src_points[3][1]) ** 2))
    height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + 
                         ((src_points[3][1] - src_points[0][1]) ** 2))
    height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + 
                          ((src_points[2][1] - src_points[1][1]) ** 2))
    
    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_left), int(height_right))
    
    # 设置目标点（矫正后的文档角点）
    dst_points = np.array([
        [0, 0],                    # 左上
        [max_width - 1, 0],        # 右上
        [max_width - 1, max_height - 1],  # 右下
        [0, max_height - 1]        # 左下
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    
    # 执行透视变换
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))
    
    return warped

# 输出目录
output_dir = Path('output/visualizations/perspective_correction')
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB以便 matplotlib 显示
    img_height, img_width, _ = img.shape
    
    # --- 获取真实标注点 --- #
    img_gt = img.copy()
    true_points = {}
    gt_warped = None
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
            
            # 使用真实标注点进行透视变换
            gt_warped = perform_perspective_transform(img, points)
            
            # 绘制四边形和点
            cv2.polylines(img_gt, [points], True, (0, 255, 0), 20)
            for point in points:
                cv2.circle(img_gt, tuple(point), 50, (255, 0, 0), -1)
    except Exception as e:
        print(f"处理真实标注时出错 {pose_label_path}: {str(e)}")
    
    # --- 获取 Pose Origin 预测点并校正 --- #
    pose_origin_warped = None
    img_pose_origin = img.copy()
    if not pd.isna(row['pose-old_pred_T_L_x']):
        pred_points_origin_640 = np.array([
            [row['pose-old_pred_T_L_x'], row['pose-old_pred_T_L_y']],
            [row['pose-old_pred_T_R_x'], row['pose-old_pred_T_R_y']],
            [row['pose-old_pred_B_R_x'], row['pose-old_pred_B_R_y']],
            [row['pose-old_pred_B_L_x'], row['pose-old_pred_B_L_y']]
        ])
        
        # 将坐标缩放到原始图像尺寸
        pred_points_origin = pred_points_origin_640.copy()
        pred_points_origin[:, 0] = pred_points_origin[:, 0] * img_width / 640
        pred_points_origin[:, 1] = pred_points_origin[:, 1] * img_height / 640
        pred_points_origin = pred_points_origin.astype(np.int32)
        
        # 使用预测点进行透视变换
        pose_origin_warped = perform_perspective_transform(img, pred_points_origin)
        
        # 绘制四边形和点
        cv2.polylines(img_pose_origin, [pred_points_origin], True, (255, 165, 0), 20)
        for point in pred_points_origin:
            cv2.circle(img_pose_origin, tuple(point), 50, (0, 0, 255), -1)
    
    # --- 获取 Pose GSCL 预测点并校正 --- #
    pose_gscl_warped = None
    img_pose_gscl = img.copy()
    if not pd.isna(row['pose-new_pred_T_L_x']):
        pred_points_gscl_640 = np.array([
            [row['pose-new_pred_T_L_x'], row['pose-new_pred_T_L_y']],
            [row['pose-new_pred_T_R_x'], row['pose-new_pred_T_R_y']],
            [row['pose-new_pred_B_R_x'], row['pose-new_pred_B_R_y']],
            [row['pose-new_pred_B_L_x'], row['pose-new_pred_B_L_y']]
        ])
        
        # 将坐标缩放到原始图像尺寸
        pred_points_gscl = pred_points_gscl_640.copy()
        pred_points_gscl[:, 0] = pred_points_gscl[:, 0] * img_width / 640
        pred_points_gscl[:, 1] = pred_points_gscl[:, 1] * img_height / 640
        pred_points_gscl = pred_points_gscl.astype(np.int32)
        
        # 使用预测点进行透视变换
        pose_gscl_warped = perform_perspective_transform(img, pred_points_gscl)
        
        # 绘制四边形和点
        cv2.polylines(img_pose_gscl, [pred_points_gscl], True, (255, 165, 0), 20)
        for point in pred_points_gscl:
            cv2.circle(img_pose_gscl, tuple(point), 50, (0, 0, 255), -1)
    
    # --- 创建并保存图表 --- #
    # 第一行：原始图像加检测框
    # 第二行：校正后的图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：原始图像加检测框
    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title('Ground Truth Detection', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_pose_origin)
    axes[0, 1].set_title(add_iou_to_title('Pose Origin Detection', row['pose-old_iou']), fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_pose_gscl)
    axes[0, 2].set_title(add_iou_to_title('GSCL Detection', row['pose-new_iou']), fontsize=10)
    axes[0, 2].axis('off')
    
    # 第二行：校正后的图像
    if gt_warped is not None:
        axes[1, 0].imshow(gt_warped)
        axes[1, 0].set_title('Ground Truth Correction', fontsize=10)
    axes[1, 0].axis('off')
    
    if pose_origin_warped is not None:
        axes[1, 1].imshow(pose_origin_warped)
        axes[1, 1].set_title('Pose Origin Correction', fontsize=10)
    axes[1, 1].axis('off')
    
    if pose_gscl_warped is not None:
        axes[1, 2].imshow(pose_gscl_warped)
        axes[1, 2].set_title('GSCL Correction', fontsize=10)
    axes[1, 2].axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_dir / f'{img_name_formatted}_perspective_correction.png', dpi=300)
    plt.close(fig)

print(f"透视校正结果已保存到 {output_dir}")
