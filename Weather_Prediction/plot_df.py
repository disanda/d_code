import matplotlib.pyplot as plt
import pandas as pd

csv_path = "mpi_saale_2021b.csv"
data_frame = pd.read_csv(csv_path)
print(data_frame.columns)

df = data_frame.copy()
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S') # 将字符串格式的时间列转换为日期时间格式
df['Date Time'] = df['Date Time'].astype(int) // 10**9  ## 将日期时间格式的时间列转换为时间戳（数字格式） 除以10**9将纳秒转换为秒


# 计算行数和列数
num_rows = 3
num_cols = 5

# 计算需要多少个图
num_plots = len(df.columns)

# 计算图表需要的数量
num_figures = num_plots // (num_rows * num_cols)
if num_plots % (num_rows * num_cols) != 0:
    num_figures += 1

# 循环遍历每个图表
for f in range(num_figures):
    # 创建一个新的图表
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 15))

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