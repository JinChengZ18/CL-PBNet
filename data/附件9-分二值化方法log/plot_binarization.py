import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 


def plot_performance_comparison(df):
    plt.figure(figsize=(12,  8))
    
    # 创建子图网格 
    ax1 = plt.subplot2grid((3,3),  (0,0), colspan=2)
    ax2 = plt.subplot2grid((3,3),  (0,2))
    ax3 = plt.subplot2grid((3,3),  (1,0), colspan=3, rowspan=2)
    
    # 准确率分布（ax1）
    sns.violinplot(x='Method',  y='Accuracy', data=df, ax=ax1, palette='viridis')
    ax1.set_title('Accuracy Distribution', fontsize=12)
    
    # 内存对比（ax2）
    mem_data = df.groupby('Method')['Memory'].mean().reset_index() 
    sns.barplot(x='Method',  y='Memory', data=mem_data, ax=ax2, palette='rocket')
    ax2.set_title('Memory Consumption', fontsize=12)
    
    # 效率-精度权衡（ax3）
    sns.scatterplot(x='Epoch',  y='Accuracy', hue='Method', 
                    data=df, ax=ax3, s=100, palette='Set2')
    ax3.annotate('CMIM  Optimal Zone', xy=(74, 92), xytext=(70, 88),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax3.set_title('Accuracy-Computation  Tradeoff', fontsize=14)
    
    plt.tight_layout() 
    plt.savefig('binarization_performance.png',  dpi=300)
    plt.show() 



# 执行可视化 
df = pd.read_csv("Binarization_Data.csv")
plot_performance_comparison(df)

