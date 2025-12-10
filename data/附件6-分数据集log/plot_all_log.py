import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.lines  import Line2D
import os


datasets = ['Satellite', 'Spambase', 'Yeast', 'Iris', 'Diabetes', 'Vehicle'] 
experiments = [f'Experiment_{i}' for i in range(1,10)]
# ======================
# 可视化模块
# ======================
plt.style.use('seaborn-v0_8-whitegrid') 
 
# 颜色映射配置
colors = plt.cm.tab10(np.linspace(0,1,9)) 
 
# 创建6个子图的画布
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten() 
 
# 遍历所有数据集
for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    ax2 = ax.twinx() 
    
    # 遍历所有实验 
    for exp_idx, exp in enumerate(experiments):
        # 读取日志数据 
        log_file = f"{dataset}-{exp}.log"
        epochs, losses, accuracies = [], [], []
        with open(log_file) as f:
            for line in f.readlines()[2:]:   # 跳过头两行 
                e, l, a = line.strip().split(', ')
                epochs.append(int(e)) 
                losses.append(float(l)) 
                accuracies.append(float(a)) 
        
        # 绘制曲线 
        ax.plot(epochs,  accuracies, color=colors[exp_idx], 
                linestyle='-', alpha=0.7, lw=1.5)
        ax2.plot(epochs,  losses, color=colors[exp_idx], 
                 linestyle='--', alpha=0.7, lw=1.5)
 
    # 坐标轴设置
    ax.set_xlabel('Epoch',  fontsize=20)
    ax.set_ylabel('Accuracy  (%)', color='tab:blue', fontsize=20)
    ax2.set_ylabel('Loss',  color='tab:red', fontsize=20)
    ax.set_title(f'{dataset}  Dataset', fontsize=30, pad=20)
    
    # 网格样式 
    ax.grid(True,  linestyle='--', alpha=0.6)
    ax2.grid(False) 
 
# 创建统一图例 
legend_elements = [
    *[Line2D([0], [0], color=colors[i], label=f'Exp{i+1}') for i in range(9)],
    Line2D([0], [0], color='black', linestyle='-', label='Accuracy'),
    Line2D([0], [0], color='black', linestyle='--', label='Loss')
]
 
plt.figlegend(handles=legend_elements,  loc='lower center', 
             ncol=6, fontsize=20, bbox_to_anchor=(0.5, -0.07))

plt.tight_layout() 
plt.savefig('All_Datasets_Visualization.png',  bbox_inches='tight')
plt.show()





