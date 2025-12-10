import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 



# 数据集基础准确率 
BASE_ACCURACY = {
    'Satellite': 92.1,
    'Spambase': 89.5,
    'Yeast': 85.7,
    'Iris': 95.2,
    'Diabetes': 88.3,
    'Vehicle': 90.6 
}
 
MODELS = ['PBNN', 'FP32', 'BinaryNet']
DATASETS = list(BASE_ACCURACY.keys()) 


def plot_results():
    """可视化结果 - 每种噪声类型单独绘图"""
    # 设置全局绘图风格
    sns.set_style("whitegrid") 
    plt.rcParams['font.family']  = 'DejaVu Sans'
    plt.rcParams['axes.labelsize']  = 12 
    plt.rcParams['axes.titlesize']  = 14
    
    # 配色方案 
    palette = {
        'PBNN': {'color': '#2ca02c', 'marker': 'o', 'ls': '-', 'lw': 2.5, 'markersize': 8},
        'FP32': {'color': '#1f77b4', 'marker': 's', 'ls': '--', 'lw': 2.5, 'markersize': 8},
        'BinaryNet': {'color': '#d62728', 'marker': '^', 'ls': '-.', 'lw': 2.5, 'markersize': 8}
    }
    
    # 为每种噪声类型创建独立图表
    for noise_type in ['Gaussian', 'Label', 'Activation']:
        # 创建新图表
        plt.figure(figsize=(10,  6.5))
        
        # 读取平均结果
        df = pd.read_csv(f'results/{noise_type}_Noise_MEAN.csv') 
        
        # 绘制每种模型的曲线
        for model in MODELS:
            subset = df[df['Model'] == model]
            style = palette[model]
            plt.plot(subset['NoiseLevel'],  subset['Accuracy'], 
                     label=f'{model}', **style)
        
        # 设置图表属性 
        plt.title(f'Robustness  to {noise_type} Noise Across Datasets', 
                 fontsize=16, pad=15, weight='bold')
        plt.xlabel('Noise  Intensity', fontsize=13, labelpad=10)
        plt.ylabel('Average  Accuracy (%)', fontsize=13, labelpad=10)
        
        # 优化坐标轴范围 
        ymin = max(df['Accuracy'].min() - 5, 40)
        ymax = min(df['Accuracy'].max() + 5, 95)
        plt.ylim(ymin,  ymax)
        plt.xlim(df['NoiseLevel'].min(),  df['NoiseLevel'].max())
        
        # 添加图例和网格
        plt.legend(loc='best',  frameon=True, shadow=True, fontsize=12)
        plt.grid(True,  linestyle='--', alpha=0.7)
        
        # 添加背景色增强可读性 
        plt.gca().set_facecolor('#f8f9fa') 
        plt.gca().spines[['top',  'right']].set_visible(False)
        
        # 添加文本标注说明噪声特性 
        noise_info = {
            'Gaussian': "Additive Gaussian noise simulates sensor errors\nand environmental interference",
            'Label': "Label noise represents annotation errors\nand misclassified training data",
            'Activation': "Activation noise models hardware faults\nand computation errors in neural units"
        }
        plt.annotate(noise_info[noise_type],  
                    xy=(0.98, 0.05), 
                    xycoords='axes fraction',
                    ha='right', va='bottom',
                    fontsize=11, 
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f8f9fa', alpha=0.8))
        
        # 优化布局并保存
        plt.tight_layout() 
        plt.savefig(f'results/{noise_type}_Noise_Robustness.png',  dpi=300, bbox_inches='tight')
        print(f"已保存 {noise_type} 噪声鲁棒性图表")
    
    # 显示所有图表（最后创建的）
    plt.show() 




# 主执行流程 
if __name__ == "__main__":
    plot_results()
