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
# 先跑出每一张图片的IoU，存在 csv，便于之后使用 matplotlib 绘制数据

import os
import json
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def calculate_iou(pred_mask, true_mask):
    # 确保两个掩码尺寸相同
    if pred_mask.shape != true_mask.shape:
        pred_mask = cv2.resize(pred_mask, (true_mask.shape[1], true_mask.shape[0]))
    
    # 二值化掩码
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)
    
    # 打印调试信息
    print(f"预测掩码形状: {pred_mask.shape}, 非零元素: {np.count_nonzero(pred_mask)}")
    print(f"真实掩码形状: {true_mask.shape}, 非零元素: {np.count_nonzero(true_mask)}")
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union > 0 else 0
    print(f"IoU: {iou}, 交集: {intersection}, 并集: {union}")
    return iou

def process_image(model, image_path, label_path, is_pose=False):
    # 加载图片并进行预测
    results = model(image_path)[0]
    print(f"\n处理图片: {image_path}")
    
    # 打印预测结果信息
    print(f"预测结果类型: {type(results)}")
    if hasattr(results, 'boxes'):
        print(f"检测到的框数量: {len(results.boxes)}")
        if len(results.boxes) > 0:
            print(f"框的置信度: {results.boxes.conf[0].item()}")
            print(f"框的坐标: {results.boxes.xyxy[0].tolist()}")
    if hasattr(results, 'masks'):
        print(f"检测到的掩码数量: {len(results.masks) if results.masks is not None else 0}")
    
    # 加载真实标注
    with open(label_path, 'r') as f:
        true_data = json.load(f)
    
    # 打印标注文件中的所有标签
    print("\n标注文件中的标签:")
    for shape in true_data['shapes']:
        print(f"标签: {shape['label']}, 类型: {shape['shape_type']}")
    
    # 获取预测掩码和预测点
    pred_mask = np.zeros((640, 640))
    pred_points = None
    
    if is_pose:
        # pose 模型：直接使用关键点预测结果
        if hasattr(results, 'keypoints') and len(results.keypoints) > 0:
            # 获取置信度最高的检测结果的关键点
            keypoints = results.keypoints[0].data.cpu().numpy()  # [1, 4, 3] shape
            if keypoints.shape[1] == 4:  # 确保有4个关键点
                # keypoints[..., :2] 获取xy坐标 (忽略置信度)
                pred_points = keypoints[0, :, :2]  # [4, 2] shape
                print(f"预测关键点坐标: {pred_points.tolist()}")
                
                # 将坐标缩放到 640x640
                pred_points[:, 0] = pred_points[:, 0] * 640 / results.orig_shape[1]
                pred_points[:, 1] = pred_points[:, 1] * 640 / results.orig_shape[0]
                pred_points = pred_points.astype(np.float32)
                print(f"缩放后关键点坐标: {pred_points.tolist()}")
                
                # 使用关键点创建四边形掩码
                cv2.fillPoly(pred_mask, [pred_points.astype(np.int32)], 1)
                print(f"预测掩码非零元素: {np.count_nonzero(pred_mask)}")
            else:
                print(f"警告：检测到的关键点数量不正确: {keypoints.shape[1]}")
        else:
            print("警告：模型没有检测到任何关键点")
    else:
        # segment 模型：从掩码获取预测点
        if hasattr(results, 'masks') and results.masks is not None:
            # 获取置信度最高的检测结果
            conf = results.boxes.conf.cpu().numpy()
            if len(conf) > 0:
                max_conf_idx = np.argmax(conf)
                mask = results.masks.data[max_conf_idx].cpu().numpy()
                
                # 将掩码转换为正确的形状
                if len(mask.shape) == 3:
                    mask = mask.squeeze()
                
                # 从掩码获取角点
                pred_points = get_document_corners_from_mask(mask, (640, 640))
                if pred_points is not None:
                    # 使用角点创建四边形掩码
                    cv2.fillPoly(pred_mask, [pred_points.astype(np.int32)], 1)
                    print(f"预测掩码非零元素: {np.count_nonzero(pred_mask)}")
        else:
            print("警告：模型没有检测到任何掩码")
    
    # 创建真实掩码和存储标注点
    true_mask = np.zeros((640, 640))
    true_points = None
    
    # 根据标注类型处理
    found_document = False
    if is_pose:
        # pose 标注格式：使用四个角点构建四边形
        corner_points = {}
        for shape in true_data['shapes']:
            if shape['shape_type'] == 'point':
                corner_points[shape['label']] = shape['points'][0]
        
        # 检查是否找到所有四个角点
        if all(corner in corner_points for corner in ['T_L', 'T_R', 'B_R', 'B_L']):
            found_document = True
            # 按顺序排列角点：左上、右上、右下、左下
            points = np.array([
                corner_points['T_L'],
                corner_points['T_R'],
                corner_points['B_R'],
                corner_points['B_L']
            ])
            print(f"原始角点: {points}")
            
            # 将坐标缩放到 640x640
            points[:, 0] = points[:, 0] * 640 / true_data['imageWidth']
            points[:, 1] = points[:, 1] * 640 / true_data['imageHeight']
            points = points.astype(np.int32)
            print(f"缩放后角点: {points}")
            
            # 创建四边形掩码
            cv2.fillPoly(true_mask, [points], 1)
            
            # 存储标注点
            true_points = {
                'T_L': points[0].tolist(),
                'T_R': points[1].tolist(),
                'B_R': points[2].tolist(),
                'B_L': points[3].tolist()
            }
    else:
        # segment 标注格式：使用多边形点构建四边形
        for shape in true_data['shapes']:
            if shape['label'] == 'paper' and shape['shape_type'] == 'polygon':
                found_document = True
                points = np.array(shape['points'])
                print(f"原始标注点: {points}")
                
                # 将坐标缩放到 640x640
                points[:, 0] = points[:, 0] * 640 / true_data['imageWidth']
                points[:, 1] = points[:, 1] * 640 / true_data['imageHeight']
                points = points.astype(np.int32)
                print(f"缩放后标注点: {points}")
                
                # 使用轮廓近似找到四个角点
                epsilon = 0.02 * cv2.arcLength(points, True)
                approx = cv2.approxPolyDP(points, epsilon, True)
                
                if len(approx) > 4:
                    # 找到轮廓边界框
                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    corners = box.astype(np.float32)
                elif len(approx) == 4:
                    # 如果正好是四个点，直接使用
                    corners = approx.reshape(4, 2).astype(np.float32)
                else:
                    # 点数小于4，使用凸包
                    hull = cv2.convexHull(points)
                    epsilon = 0.1 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    
                    if len(approx) < 4:
                        print(f"无法找到足够的角点，只有 {len(approx)} 个")
                        continue
                    
                    # 保留前4个点
                    corners = approx[:4, 0, :].astype(np.float32)
                
                # 确保点是按顺序排列的
                # 首先按y坐标排序（从上到下）
                corners = corners[np.argsort(corners[:, 1])]
                
                # 然后上面两个点按x坐标排序
                top_points = corners[:2]
                top_points = top_points[np.argsort(top_points[:, 0])]
                
                # 下面两个点按x坐标排序
                bottom_points = corners[2:]
                bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
                
                # 组合点（左上、右上、右下、左下）
                corners = np.array([
                    top_points[0],     # 左上
                    top_points[1],     # 右上
                    bottom_points[1],  # 右下
                    bottom_points[0]   # 左下
                ], dtype=np.float32)
                
                # 创建四边形掩码
                cv2.fillPoly(true_mask, [corners.astype(np.int32)], 1)
                
                # 存储标注点
                true_points = {
                    'T_L': corners[0].tolist(),
                    'T_R': corners[1].tolist(),
                    'B_R': corners[2].tolist(),
                    'B_L': corners[3].tolist()
                }
                break
    
    if not found_document:
        print(f"警告：标注文件中没有找到{'四个角点' if is_pose else '文档多边形'}")
    
    # 保存掩码图像用于调试
    # cv2.imwrite(f'debug_pred_mask_{os.path.basename(image_path)}.png', (pred_mask * 255).astype(np.uint8))
    # cv2.imwrite(f'debug_true_mask_{os.path.basename(image_path)}.png', (true_mask * 255).astype(np.uint8))
    
    return calculate_iou(pred_mask, true_mask), pred_points, true_points

def get_document_corners_from_mask(mask, img_shape):
    """
    从分割掩码中获取文档的四个角点
    
    Args:
        mask: 分割掩码，numpy数组，值范围0-1
        img_shape: 原始图像的形状 (height, width)
    
    Returns:
        corners: 四个角点的坐标，按左上、右上、右下、左下顺序排列的numpy数组，shape为(4, 2)
        如果无法找到有效的角点，返回None
    """
    try:
        # 将掩码转换为二值图像
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 调整大小以匹配原始图像
        if mask.shape[:2] != img_shape[:2]:
            mask = cv2.resize(mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未找到有效轮廓")
            return None
        
        # 找到最大的轮廓（假设是文档）
        max_contour = max(contours, key=cv2.contourArea)
        
        # 使用轮廓近似以找到文档的顶点
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # 如果近似轮廓点太多，尝试找到四个最主要的点
        if len(approx) > 4:
            # 找到轮廓边界框
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            src_points = box.astype(np.float32)
        elif len(approx) == 4:
            # 如果正好是四个点，直接使用
            src_points = approx.reshape(4, 2).astype(np.float32)
        else:
            # 点数小于4，使用凸包
            hull = cv2.convexHull(max_contour)
            epsilon = 0.1 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) < 4:
                print(f"无法找到足够的角点，只有 {len(approx)} 个")
                return None
            
            # 保留前4个点
            src_points = approx[:4, 0, :].astype(np.float32)
        
        # 确保点是按顺序排列的
        # 首先按y坐标排序（从上到下）
        src_points = src_points[np.argsort(src_points[:, 1])]
        
        # 然后上面两个点按x坐标排序
        top_points = src_points[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]
        
        # 下面两个点按x坐标排序
        bottom_points = src_points[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        
        # 组合点（左上、右上、右下、左下）
        corners = np.array([
            top_points[0],     # 左上
            top_points[1],     # 右上
            bottom_points[1],  # 右下
            bottom_points[0]   # 左下
        ], dtype=np.float32)
        
        return corners
        
    except Exception as e:
        print(f"获取文档角点时出错: {str(e)}")
        return None

def perform_document_correction(self, img, result):
    """执行文档校正，基于分割掩码"""
    # 检查是否有检测结果
    if len(result.boxes) == 0 or not hasattr(result, 'masks') or result.masks is None:
        print("未检测到文档或分割掩码")
        return
    
    # 获取置信度最高的检测结果
    conf = result.boxes.conf.cpu().numpy()
    if len(conf) == 0:
        return
        
    max_conf_idx = np.argmax(conf)
    
    try:
        # 获取掩码数据
        mask = result.masks.data[max_conf_idx].cpu().numpy()
        
        # 将掩码转换为正确的形状
        if len(mask.shape) == 3:
            mask = mask.squeeze()  # 移除额外的维度
        
        # 获取文档角点
        corners = get_document_corners_from_mask(mask, img.shape)
        if corners is None:
            return
        
        # 计算目标文档的大小
        width_top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + 
                           ((corners[1][1] - corners[0][1]) ** 2))
        width_bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + 
                              ((corners[2][1] - corners[3][1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))
        
        height_left = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + 
                             ((corners[3][1] - corners[0][1]) ** 2))
        height_right = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + 
                              ((corners[2][1] - corners[1][1]) ** 2))
        max_height = max(int(height_left), int(height_right))
        
        # 设置目标点
        dst_points = np.array([
            [0, 0],                  # 左上
            [max_width - 1, 0],      # 右上
            [max_width - 1, max_height - 1],  # 右下
            [0, max_height - 1]      # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners, dst_points)
        
        # 执行透视变换
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        self.warped_image = warped.copy()  # 保存校正后的图像
        
        # 显示校正后的图像
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        self.display_image(warped_rgb, self.processed_canvas)
    except Exception as e:
        print(f"执行文档校正时出错: {str(e)}")

def main():
    # 加载模型
    models = {
        'pose-old': load_model('pose-old-best.pt'),
        'pose-new': load_model('pose-new-best.pt'),
        'segment': load_model('segment-best.pt')
    }
    
    # 准备数据存储
    results = []
    
    # 遍历所有图片
    image_dir = Path('datasets/images')
    pose_label_dir = Path('datasets/pose-labels')
    segment_label_dir = Path('datasets/segment-labels')
    
    for background in ['complex', 'dark_pure', 'shadow_light', 'white_pure', 'wood_pure']:
        for distance in ['35cm', '45cm', '55cm', '右下角正对手持', '左下角正对手持', '底边正对手持']:
            current_dir = image_dir / background / distance
            if not current_dir.exists():
                continue
                
            for img_file in tqdm(list(current_dir.glob('*.jpg')), desc=f'Processing {background}/{distance}'):
                img_name = img_file.stem
                pose_label = pose_label_dir / f'{img_name}.json'
                segment_label = segment_label_dir / f'{img_name}.json'
                
                if not (pose_label.exists() and segment_label.exists()):
                    continue
                
                result = {
                    'background': background,
                    'distance': distance,
                    'image': img_name
                }
                
                # 计算每个模型的 IoU
                for model_name, model in models.items():
                    print(f"\n使用模型: {model_name}")
                    label_path = pose_label if 'pose' in model_name else segment_label
                    iou, pred_points, true_points = process_image(model, str(img_file), str(label_path), is_pose='pose' in model_name)
                    result[f'{model_name}_iou'] = iou
                    
                    # 存储预测点和标注点
                    if pred_points is not None:
                        # 预测点是numpy数组，按左上、右上、右下、左下顺序
                        result[f'{model_name}_pred_T_L_x'] = pred_points[0][0]
                        result[f'{model_name}_pred_T_L_y'] = pred_points[0][1]
                        result[f'{model_name}_pred_T_R_x'] = pred_points[1][0]
                        result[f'{model_name}_pred_T_R_y'] = pred_points[1][1]
                        result[f'{model_name}_pred_B_R_x'] = pred_points[2][0]
                        result[f'{model_name}_pred_B_R_y'] = pred_points[2][1]
                        result[f'{model_name}_pred_B_L_x'] = pred_points[3][0]
                        result[f'{model_name}_pred_B_L_y'] = pred_points[3][1]
                    
                    if true_points is not None:
                        # 标注点是字典，包含T_L, T_R, B_R, B_L
                        for corner, coords in true_points.items():
                            result[f'{model_name}_true_{corner}_x'] = coords[0]
                            result[f'{model_name}_true_{corner}_y'] = coords[1]
                
                results.append(result)
    
    # 保存结果到 CSV
    df = pd.DataFrame(results)
    df.to_csv('experiment_results.csv', index=False)
    print("实验完成，结果已保存到 experiment_results.csv")

if __name__ == '__main__':
    main()