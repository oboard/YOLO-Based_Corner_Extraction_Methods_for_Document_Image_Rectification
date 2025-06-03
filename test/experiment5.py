import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于 macOS
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('experiment_results.csv')

# 定义要分析的背景类型
backgrounds = ['complex', 'shadow_light', 'white_pure', 'wood_pure']

# 过滤出这四个背景类型的数据
df_filtered = df[df['background'].isin(backgrounds)]

# 计算三个模型在这些背景下的平均IoU
pose_origin_ious = []
gscl_ious = []
segment_ious = []

for bg in backgrounds:
    bg_data = df_filtered[df_filtered['background'] == bg]
    pose_origin_ious.append(bg_data['pose-old_iou'].mean() if not bg_data.empty else 0)
    gscl_ious.append(bg_data['pose-new_iou'].mean() if not bg_data.empty else 0)
    segment_ious.append(bg_data['segment_iou'].mean() if not bg_data.empty else 0)

# 设置柱状图的位置和宽度
bar_width = 0.25
index = np.arange(len(backgrounds))

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制柱状图
bar1 = ax.bar(index - bar_width, pose_origin_ious, bar_width, label='YOLO Pose Origin')
bar2 = ax.bar(index, gscl_ious, bar_width, label='YOLO Pose-GSCL')
bar3 = ax.bar(index + bar_width, segment_ious, bar_width, label='YOLO Segment')

# 设置标题和标签
ax.set_title('Model Performance Comparison (IoU)')
ax.set_xlabel('Type of Background')
ax.set_ylabel('Average IoU')
ax.set_xticks(index)
ax.set_xticklabels(backgrounds, rotation=0)
ax.set_ylim(0.5, 1.2)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7, axis='y')

# 在每个柱子上方添加具体数值
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

add_value_labels(bar1)
add_value_labels(bar2)
add_value_labels(bar3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('three_models_comparison.png', dpi=300)

print("图表已保存为 'three_models_comparison.png'")
