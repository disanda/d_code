import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

extended = './nasdaq100/extended/extended_non_padding.csv' # (8993, 10)
small = './nasdaq100/small/nasdaq100_padding.csv' # (40560,82)
full = './nasdaq100/full/full_non_padding.csv'  # (74501, 105)
sperate_aal_1 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-26.csv'
sperate_aal_2 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-27.csv'
sperate_aal_3 = './nasdaq100/full/stock_data_GOOGLE/AAL_2016-07-28.csv'

#数据清洗，去除NaN数据，用邻近均值做填充(padding)
df = pd.read_csv(full) # nrows=3 

# 显示3日内股票特征的df
# df1 = pd.read_csv(sperate_aal_1)
# df2 = pd.read_csv(sperate_aal_2)
# df3 = pd.read_csv(sperate_aal_3)
# df = pd.concat([df1,df2,df3],axis=0)

columns = df.columns
print(df.shape)
print(df.columns)

print(df.iloc[:5,:8])
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] # 对列的NaN输出坐标

data = df.to_numpy()
for col in range(data.shape[1]):
    nans, x = nan_helper(data[:,col])
    data[nans,col] = np.interp(x(nans),x(~nans),data[~nans,col])

df = pd.DataFrame(data,columns = columns)
print(df.iloc[:5,:8].round(4)) # .round(4)


# ## --------- 计算线性相关系数 ----------
# #correlations = df.iloc[:, :-1].corrwith(df['NDX']) # Pearson, NDX就是 Nasdaq-100指数
# #correlations = df.corr(method='pearson')['NDX'].iloc[:-1] # Pearson
# #correlations = df.corr(method='spearman')['NDX'].iloc[:-1] # Spearman
# correlations = df.corr(method='kendall')['NDX'].iloc[:-1] # Kendall

# # 可视化
# plt.figure(figsize=(33, 11))
# #bars = correlations.plot(kind='bar', color='green') #blue, green, orange
# bars = plt.bar(range(len(correlations)), correlations, color='orange', alpha=0.7)
# #plt.title('Pearson Correlation with Index Volatility')  # Pearson
# #plt.title('Spearman Correlation with Index Volatility') # Spearman
# plt.title('Kendall Correlation with Index Volatility') # Kendall
# plt.xlabel('Stocks')
# plt.ylabel('Correlation')
# #plt.xticks(rotation=45)

# # 设置x轴标签为特征的具体名称和数字顺序id
# feature_names = df.columns[:-1]
# plt.xticks(range(len(correlations)), [f"{i}: {name}" for i, name in enumerate(feature_names)], rotation=45)

# # 在柱形上方标记特征的索引号
# for i, bar in enumerate(bars):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(i), ha='center', va='bottom')

# plt.grid(True)
# plt.tight_layout()
# #plt.savefig(f'Pearson Correlation.png') # 保存图像
# #plt.savefig(f'Spearman Correlation.png') # 保存图像
# plt.savefig(f'Kendall Correlation.png') # 保存图像
# plt.close() # 关闭图表以释放内存

## --------- 可视化互信息 ----------
# 将指数波动列从DataFrame中分离出来作为目标变量
# y = df['NDX']
# X = df.drop(columns=['NDX'])

# # 计算每个特征与目标之间的互信息
# mi_scores = mutual_info_regression(X, y)

# plt.figure(figsize=(33, 9))
# bars = plt.bar(range(len(mi_scores)), mi_scores, color='red', alpha=0.7)
# plt.title('Mutual Information between Features and Index Volatility')
# plt.xlabel('Features')
# plt.ylabel('Mutual Information')
# #plt.xticks(rotation=45)
# #plt.xticks(range(len(mi_scores)), X.columns, rotation=45)

# feature_names = X.columns
# plt.xticks(range(len(mi_scores)), [f"{i}: {name}" for i, name in enumerate(feature_names)], rotation=45) # 设置x轴标签为特征的具体名称和数字顺序id

# for i, bar in enumerate(bars): 
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(i), ha='center', va='bottom') # 在柱形上方标记特征的索引号

# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'Mutual information.png') # 保存图像
# plt.close() # 关闭图表以释放内存

# ## --------- xgboost----------
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # 将数据拆分为训练集和测试集
# y = df['NDX']
# X = df.drop(columns=['NDX'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # 使用XGBoost训练模型
# model = xgb.XGBRegressor()
# model.fit(X_train, y_train)

# # 对测试集进行预测
# y_pred = model.predict(X_test)

# # 计算均方根误差（Root Mean Squared Error）
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE:", rmse)

# # 可视化特征重要性
# plt.figure(figsize=(21, 8))
# xgb.plot_importance(model, max_num_features=30, height=0.7)
# plt.title('XGBoost Feature Importance')
# plt.xlabel('F Score')
# plt.ylabel('Features')
# plt.savefig(f'XGBoost Feature Importance.png') # 保存图像
# plt.close() # 关闭图表以释放内存

## ---------PCA----------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数据预处理：标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 使用PCA进行降维
pca = PCA()
pca.fit(scaled_data)


# 反转特征顺序
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cumulative_variance_reversed = cumulative_variance[::-1]

# 绘制每个主成分的方差解释比例
plt.figure(figsize=(33, 15))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')

# 标注原始特征的索引号
for i in range(len(df.columns[:104])):
    #label = f'{i}: {df.columns[i]}'  # 结合索引号和名称
    label = f'{len(df.columns) - 1 - i}: {df.columns[::-1][i]}'  # 结合索引号和名称，注意反转特征顺序并调整索引号
    if i % 2 == 0:
        plt.text(i, np.cumsum(pca.explained_variance_ratio_)[i], label, fontsize=7, ha='right', va='bottom', alpha=0.99)
    else:
        plt.text(i, np.cumsum(pca.explained_variance_ratio_)[i], label, fontsize=7, ha='left', va='top', alpha=0.99)

plt.grid(True)
plt.savefig(f'Explained Variance by Principal Components.png') # 保存图像
plt.close() # 关闭图表以释放内存

# 设定方差解释比例阈值
explained_variance_threshold = 0.95
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > explained_variance_threshold) + 1

# 选择前n_components个主成分
selected_components = pca.components_[:n_components, :]

# 获取与主成分相关的原始特征及其索引号
correlated_features = []
for component in selected_components:
    feature_indices = np.argsort(np.abs(component))[-5:]  # 假设选择与每个主成分相关性最高的5个特征
    correlated_features.append([(f, df.columns[f]) for f in feature_indices])

# 可视化与主成分相关的原始特征
plt.figure(figsize=(12, 8))  # 调整图像大小
for i, component in enumerate(selected_components):
    feature_indices = np.arange(i*5, (i+1)*5)
    feature_names = [f'{f[0]}: {f[1]}' for f in correlated_features[i]]
    plt.barh(feature_indices, np.abs(component[np.argsort(np.abs(component))[-5:]]), height=0.4, label=f'Component {i+1}')
plt.xlabel('Absolute Correlation Coefficient')
yticks_labels = [f'{i+1}: {f[0]} - {f[1]}' for i, sublist in enumerate(correlated_features) for f in sublist]  # 原始特征的索引号和名称
plt.yticks(np.arange(n_components*5), yticks_labels, fontsize=8)  # 在刻度上显示原始特征的索引号和名称
plt.title('Correlation of Original Features with Principal Components')
plt.legend()
plt.tight_layout()
plt.savefig(f'PCA Visualization.png') # 保存图像
plt.close() # 关闭图表以释放内存

