import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('experiment_results.csv')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于 macOS
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(15, 10))

# 为每个距离创建一个子图
distances = df['distance'].unique()
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# 为每个距离绘制柱状图
for idx, distance in enumerate(distances):
    # 获取当前距离的数据
    distance_data = df[df['distance'] == distance]
    
    # 获取所有背景类型
    backgrounds = ['complex', 'shadow_light', 'white_pure', 'wood_pure']
    # backgrounds = ['complex', 'dark_pure', 'shadow_light', 'white_pure', 'wood_pure']
    
    # 计算每个模型在不同背景下的平均IoU
    pose_old_ious = []
    pose_new_ious = []
    segment_ious = []
    
    for bg in backgrounds:
        bg_data = distance_data[distance_data['background'] == bg]
        pose_old_ious.append(bg_data['pose-old_iou'].mean() if not bg_data.empty else 0)
        pose_new_ious.append(bg_data['pose-new_iou'].mean() if not bg_data.empty else 0)
        segment_ious.append(bg_data['segment_iou'].mean() if not bg_data.empty else 0)
    
    # 设置柱状图的宽度和位置
    bar_width = 0.25
    index = np.arange(len(backgrounds))
    
    # 绘制柱状图
    ax = axes[idx]
    ax.bar(index - bar_width, pose_old_ious, bar_width, label='Pose-Old')
    ax.bar(index, pose_new_ious, bar_width, label='Pose-New')
    ax.bar(index + bar_width, segment_ious, bar_width, label='Segment')
    
    # 设置标题和标签
    ax.set_title(f'距离: {distance}')
    ax.set_xlabel('背景类型')
    ax.set_ylabel('平均IoU')
    ax.set_xticks(index)
    ax.set_xticklabels(backgrounds, rotation=45)
    ax.legend()
    
    # 设置y轴范围
    ax.set_ylim(0, 1.1)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
plt.savefig('iou_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建总体平均IoU的柱状图
plt.figure(figsize=(10, 6))

# 计算每个模型的总体平均IoU
models = ['pose-old', 'pose-new', 'segment']
mean_ious = [df[f'{model}_iou'].mean() for model in models]

# 绘制柱状图
bars = plt.bar(models, mean_ious)

# 在柱子上方显示具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

# 设置标题和标签
plt.title('各模型总体平均IoU')
plt.xlabel('模型')
plt.ylabel('平均IoU')
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('overall_iou.png', dpi=300, bbox_inches='tight')
plt.close()

print("图表已保存为 'iou_comparison.png' 和 'overall_iou.png'")