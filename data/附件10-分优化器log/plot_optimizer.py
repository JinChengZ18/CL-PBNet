import matplotlib.pyplot  as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 


def plot_optimizer_comparison(df):
    plt.figure(figsize=(12,  8))
    
    # 准确率分布与收敛速度联合图 
    plt.subplot(2,2,1) 
    sns.boxplot(x='Optimizer',  y='Accuracy', data=df, palette='Set2')
    plt.title('Accuracy Distribution')
    
    plt.subplot(2,2,2) 
    sns.violinplot(x='Optimizer',  y='Epoch', data=df, palette='viridis')
    plt.title('Convergence Epoch')
    
    # 内存-准确率权衡图 
    plt.subplot(2,1,2) 
    sns.scatterplot(x='Memory',  y='Accuracy', hue='Optimizer', 
                    data=df, s=100, palette='deep')
    plt.annotate('Optimal  Zone', xy=(2.8, 78), xytext=(2.5, 75),
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.title('Memory-Accuracy Tradeoff')
    
    plt.tight_layout() 
    plt.savefig('optimizer_performance.png',  dpi=300)
    plt.show() 



# 执行可视化 
optimizer_df = pd.read_csv("Optimizer_Data.csv")
plot_optimizer_comparison(optimizer_df)

