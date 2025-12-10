import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# 可视化代码 
def plot_robustness_curves(df, noise_type, dataset):
    plt.figure(figsize=(10,6)) 
    for model in ['PBNN', 'FP32', 'BinaryNet', 'XNOR-Net']:
        subset = df[(df['Model']==model) & (df['Dataset']==dataset)]
        plt.plot(subset['NoiseLevel'],  subset['Accuracy'], 
                 marker='o', linestyle='--', label=model)
    
    plt.xlabel(f'{noise_type}  Intensity')
    plt.ylabel('Accuracy(%)') 
    plt.title(f'Robustness  to {noise_type} on {dataset}')
    plt.legend() 
    plt.grid(True) 
    # 生成包含dataset和noise_type的文件名
    clean_noise = noise_type.replace(' ', '_').replace('-', '')
    filename = f"robustness_{dataset}_{clean_noise}.png"
    plt.savefig(filename,  dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()   # 关闭图形释放内存



df = pd.read_csv("Robustness_data_exp1.CSV")
plot_robustness_curves(df[df['Dataset']=='Satellite'], 'Gaussian Noise', 'Satellite')
plot_robustness_curves(df[df['Dataset']=='Spambase'], 'Gaussian Noise', 'Spambase')
plot_robustness_curves(df[df['Dataset']=='Yeast'], 'Gaussian Noise', 'Yeast')
plot_robustness_curves(df[df['Dataset']=='Iris'], 'Gaussian Noise', 'Iris')
plot_robustness_curves(df[df['Dataset']=='Diabetes'], 'Gaussian Noise', 'Diabetes')
plot_robustness_curves(df[df['Dataset']=='Vehicle'], 'Gaussian Noise', 'Vehicle')