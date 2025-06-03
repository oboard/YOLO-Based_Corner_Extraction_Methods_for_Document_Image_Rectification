import os
from ultralytics import YOLO

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 yaml ，四个关键点
yaml_content = f"""
train: {os.path.join(current_dir, 'datasets/train')}
val: {os.path.join(current_dir, 'datasets/val')}

nc: 1
names: ['paper']
"""

# 保存配置文件
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

# 使用自定义模型进行训练
model = YOLO('yolo11n-seg.pt')
model.train(
    data="data.yaml",
    batch=32,
    workers=16,
    exist_ok=True,
    plots=True,
    epochs=100,
    device="0",
    imgsz=640,
    cache=True,
    val=True,
    amp=True,
    optimizer="AdamW",
)
