import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

extended = './nasdaq100/extended/extended_non_padding.csv' # (8993, 10)
small = './nasdaq100/small/nasdaq100_padding.csv' # (40560,82)
full = './nasdaq100/full/full_non_padding.csv'  # (74501, 105)
sperate_aal_1 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-26.csv'
sperate_aal_2 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-27.csv'
sperate_aal_3 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-28.csv'

#数据清洗，去除NaN数据，用邻近均值做填充(padding)
#df = pd.read_csv(full) # nrows=3
df1 = pd.read_csv(sperate_aal_1)
df2 = pd.read_csv(sperate_aal_2)
df3 = pd.read_csv(sperate_aal_3)
df = pd.concat([df1,df2,df3],axis=0)

columns = df.columns
print(df.shape)
print(df.columns)

print(df.iloc[:5,:8])
def nan_helper(y):
    """处理 NaN 值的索引和逻辑索引的辅助函数。
    输入：
    - y，1D NumPy 数组，可能包含 NaN 值
    输出：
    - nans，NaN 的逻辑索引
    - index，一个函数，具有签名 indices = index(logical_indices)，用于将 NaN 的逻辑索引转换为“等效”的索引
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

data = df.to_numpy()
for col in range(data.shape[1]):
    nans, x = nan_helper(data[:,col])
    data[nans,col] = np.interp(x(nans),x(~nans),data[~nans,col])

df = pd.DataFrame(data,columns = columns)
print(df.iloc[:5,:8].round(4)) # .round(4)


# 数据可视化 
# 计算行数和列数
num_rows = 1
num_cols = 7

# 计算需要多少个图
num_plots = len(df.columns)

# 计算图表需要的数量
num_figures = num_plots // (num_rows * num_cols)
if num_plots % (num_rows * num_cols) != 0:
    num_figures += 1

# 循环遍历每个图表
for f in range(num_figures):
    # 创建一个新的图表
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 4)) #(25, 13)

    # 将图表展平以便于索引
    axes = axes.flatten()

    # 计算图表在DataFrame中的开始和结束列索引
    start_idx = f * num_rows * num_cols
    end_idx = min((f + 1) * num_rows * num_cols, num_plots)

    # 循环遍历每列并绘制相应的图表
    for i, col in enumerate(df.columns[start_idx:end_idx]):
        ax = axes[i]  # 获取当前的轴
        ax.plot(df.index, df[col])  # 绘制二维图
        ax.set_title(col)  # 设置图表标题

    # 隐藏多余的子图
    for i in range(end_idx - start_idx, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # 保存图像
    plt.savefig(f'plots_{f}.png')

    # 关闭图表以释放内存
    plt.close()