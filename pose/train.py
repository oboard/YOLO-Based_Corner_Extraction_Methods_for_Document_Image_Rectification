import os
from custom_yolo import CustomYOLO

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 yaml ，四个关键点
yaml_content = f"""
train: {os.path.join(current_dir, 'datasets/train')}
val: {os.path.join(current_dir, 'datasets/val')}

nc: 1
names: ['paper']

# Keypoints
kpt_shape: [4, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [1, 0, 3, 2]
"""

# 保存配置文件
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

# 使用自定义模型进行训练
model = CustomYOLO('yolo11l-pose.pt')
model.train(
    data="data.yaml",
    batch=48,
    workers=24,
    exist_ok=False,
    plots=True,
    epochs=100,
    device=[0],
    imgsz=640,
    cache=True,
    val=True,
    # 关键点相关参数
    pose=12.0,  # 关键点损失权重
    kobj=1.0,  # 关键点对象损失权重
    # 优化参数
    optimizer='AdamW',  # 使用AdamW优化器
    lr0=0.001,  # 初始学习率
    lrf=0.01,  # 最终学习率
    momentum=0.937,  # SGD动量
    weight_decay=0.0005,  # 权重衰减
    warmup_epochs=3,  # 预热轮数
    warmup_momentum=0.8,  # 预热动量
    warmup_bias_lr=0.1,  # 预热偏置学习率
    # 数据增强
    mosaic=0.5,  # 马赛克增强概率
    mixup=0.5,  # mixup增强概率
    copy_paste=0.3,  # 复制粘贴增强概率
    # 其他参数
    close_mosaic=10,  # 最后10个epoch关闭马赛克增强
    resume=False,  # 不恢复训练
    amp=True  # 使用混合精度训练
)
