import torch
import torch.nn as nn
from ultralytics import YOLO
import math

class CustomYOLO(YOLO):
    def __init__(self, model='yolo11n-pose.pt'):
        super().__init__(model)
        
    def _setup_loss(self):
        """设置损失函数，添加几何约束"""
        super()._setup_loss()
        
        # 添加几何约束损失
        self.geometric_loss = GSCLoss()
        
    def criterion(self, preds, batch):
        """重写损失函数计算"""
        try:
            # 计算原始损失
            loss = super().criterion(preds, batch)
            
            # 获取关键点预测
            if isinstance(preds, (list, tuple)) and len(preds) > 1:
                pred_kpts = preds[1]  # 假设关键点预测在第二个输出
            else:
                # 如果preds不是列表或元组，尝试从batch中获取预测的关键点
                pred_kpts = batch.get('pred_kpts', None)
            
            gt_kpts = batch.get('keypoints', None)
            
            if pred_kpts is not None and gt_kpts is not None:
                # 计算几何约束损失
                geometric_loss = self.geometric_loss(pred_kpts, gt_kpts)
                
                # 将几何约束损失添加到总损失中
                loss['geometric'] = geometric_loss
                loss['total'] += geometric_loss * 0.1  # 几何约束损失权重
            
            return loss
        except Exception as e:
            print(f"损失函数计算错误: {str(e)}")
            # 在异常情况下，返回原始损失
            if 'loss' in locals():
                return loss
            else:
                return super().criterion(preds, batch)
      
class GSCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_kpts, gt_kpts):
        """GSCL损失"""
        try:
            # 确保输入格式正确
            if pred_kpts is None or gt_kpts is None:
                return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 只考虑可见的关键点
            pred_visible = pred_kpts[..., 2] > 0.5 if pred_kpts.size(-1) > 2 else torch.ones_like(pred_kpts[..., 0], dtype=torch.bool)
            gt_visible = gt_kpts[..., 2] > 0.5 if gt_kpts.size(-1) > 2 else torch.ones_like(gt_kpts[..., 0], dtype=torch.bool)
            visible = pred_visible & gt_visible
            
            # 获取批次大小
            batch_size = pred_kpts.shape[0]
            
            # 初始化损失
            total_batch_loss = torch.tensor(0.0, device=pred_kpts.device)
            valid_samples = 0
            
            # 对每个样本单独处理
            for i in range(batch_size):
                # 获取当前样本的可见关键点坐标
                pred_coords_i = pred_kpts[i, visible[i], :2]
                gt_coords_i = gt_kpts[i, visible[i], :2]
                
                # 如果关键点数量不足4个，跳过该样本
                if len(pred_coords_i) < 4:
                    continue
                
                # 确保是4个关键点
                if len(pred_coords_i) > 4:
                    pred_coords_i = pred_coords_i[:4]
                    gt_coords_i = gt_coords_i[:4]
                
                # 1. 形状约束
                L_shape = self.calc_L_shape(pred_coords_i, gt_coords_i)
                # 2. 边缘约束
                L_Edge = self.calc_L_Edge(pred_coords_i)
                # 3. 位置约束
                L_position = self.calc_L_position(pred_coords_i, gt_coords_i)
                
                # 4. GSCL损失
                L_GSCL = (
                    0.4 * L_shape + 
                    0.3 * L_Edge + 
                    0.3 * L_position
                )
                
                total_batch_loss += L_GSCL
                valid_samples += 1
            
            # 计算批次平均损失
            if valid_samples > 0:
                total_batch_loss = total_batch_loss / valid_samples
            
            return total_batch_loss
        
        except Exception as e:
            print(f"GSCL损失计算错误: {str(e)}")
            return torch.tensor(0.0, device=pred_kpts.device)
        
    def calc_L_shape(self, pred_coords, gt_coords):
        """形状约束（交比）"""
        try:
            # 确保有4个点
            if len(pred_coords) < 4 or len(gt_coords) < 4:
                return torch.tensor(0.0, device=pred_coords.device)
            # 排序点，确保顺时针或逆时针排列
            p1, p2, p3, p4 = pred_coords[0], pred_coords[1], pred_coords[2], pred_coords[3]
            g1, g2, g3, g4 = gt_coords[0], gt_coords[1], gt_coords[2], gt_coords[3]
            # 计算对角线长度
            d13 = torch.norm(p3 - p1)
            d24 = torch.norm(p4 - p2)
            gd13 = torch.norm(g3 - g1)
            gd24 = torch.norm(g4 - g2)
            # 计算交比
            pred_cross_ratio = d13 / (d24 + 1e-6)
            gt_cross_ratio = gd13 / (gd24 + 1e-6)
            # 损失
            L_shape = torch.abs(pred_cross_ratio - gt_cross_ratio).mean()
            return L_shape
        except Exception as e:
            print(f"L_shape计算错误: {str(e)}")
            return torch.tensor(1.0, device=pred_coords.device)
        
    def calc_L_Edge(self, coords):
        """边缘约束（平滑度）"""
        try:
            if len(coords) < 4:
                return torch.tensor(0.0, device=coords.device)
            p1, p2, p3, p4 = coords[0], coords[1], coords[2], coords[3]
            v12 = p2 - p1
            v23 = p3 - p2
            v34 = p4 - p3
            v41 = p1 - p4
            l12 = torch.norm(v12)
            l23 = torch.norm(v23)
            l34 = torch.norm(v34)
            l41 = torch.norm(v41)
            parallelism1 = torch.abs(l12 - l34) / (l12 + l34 + 1e-6)
            parallelism2 = torch.abs(l23 - l41) / (l23 + l41 + 1e-6)
            dot1 = torch.abs(torch.sum(v12 * v41) / (l12 * l41 + 1e-6))
            dot2 = torch.abs(torch.sum(v12 * v23) / (l12 * l23 + 1e-6))
            dot3 = torch.abs(torch.sum(v23 * v34) / (l23 * l34 + 1e-6))
            dot4 = torch.abs(torch.sum(v34 * v41) / (l34 * l41 + 1e-6))
            perpendicular_loss = (dot1 + dot2 + dot3 + dot4) / 4
            L_Edge = (parallelism1 + parallelism2) / 2 + perpendicular_loss
            return L_Edge
        except Exception as e:
            print(f"L_Edge计算错误: {str(e)}")
            return torch.tensor(0.0, device=coords.device)
        
    def calc_L_position(self, pred, gt):
        """位置约束"""
        try:
            if len(pred) < 4 or len(gt) < 4:
                return torch.tensor(0.0, device=pred.device)
            dist = torch.norm(pred - gt, dim=1).mean()
            pred_area = self.compute_quadrilateral_area(pred)
            gt_area = self.compute_quadrilateral_area(gt)
            area_ratio = torch.abs(pred_area - gt_area) / (gt_area + 1e-6)
            pred_center = torch.mean(pred, dim=0)
            gt_center = torch.mean(gt, dim=0)
            pred_rel = pred - pred_center
            gt_rel = gt - gt_center
            rel_pos_consistency = torch.norm(pred_rel - gt_rel, dim=1).mean()
            L_position = 0.4 * dist + 0.3 * area_ratio + 0.3 * rel_pos_consistency
            return L_position
        except Exception as e:
            print(f"L_position计算错误: {str(e)}")
            return torch.tensor(0.0, device=pred.device)
    
    def compute_quadrilateral_area(self, points):
        """计算四边形面积"""
        # 确保有4个点
        if len(points) < 4:
            return torch.tensor(0.0, device=points.device)
        
        # 假设点的顺序是左上、右上、右下、左下
        p1, p2, p3, p4 = points[0], points[1], points[2], points[3]
        
        # 使用叉积计算面积
        # 将四边形分成两个三角形
        area1 = 0.5 * torch.abs(torch.cross(p2 - p1, p3 - p1))
        area2 = 0.5 * torch.abs(torch.cross(p3 - p1, p4 - p1))
        
        return area1 + area2