import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import models
from torch.autograd import Variable
import torch.nn.functional as F


full = './nasdaq100/full/full_non_padding.csv'  # (74501, 105)
df = pd.read_csv(full) # nrows=3
columns = df.columns
print(df.shape)
print(df.columns)
print(df.iloc[:5,:8])

#数据清洗，去除NaN数据，用邻近均值做填充(padding)
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
data = df.to_numpy()
for col in range(data.shape[1]):
    nans, x = nan_helper(data[:,col])
    data[nans,col] = np.interp(x(nans),x(~nans),data[~nans,col])
df = pd.DataFrame(data,columns = columns)
print(df.iloc[:5,:8])

# 超参数
input_size = 32
hidden_size = 128
output_size = 1
num_layers = 2
num_epochs = 20
learning_rate = 1e-6 # 1e-3开始调整(0.001)
batch_size = 8
sequence_length = 4  # 输入9个特征，预测第10个特征
model_type ='GRU' # GRU

#这里需要保留前64个特征和最后一个指数特征
data1 = df.iloc[:,:(input_size-1)] # 31
data2 = df.iloc[:,-1]  # 1个指数
data = pd.concat([data1,data2],axis=1) #按列拼接
print(data.shape)
print(data.iloc[:5,:8])

# 标准化特征
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
#print(data_scaled)
print(data_scaled[:5,:8])

# 创建序列数据
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length-1])  # 前n-1个时间步的特征
    y.append(data_scaled[i+sequence_length-1, -1])  #  第10个时间步的最后一个特征'NDX'作为目标

X = np.array(X)
y = np.array(y)

# 转换为 PyTorch 张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 默认为 CPU
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 划分训练集和测试集
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset = TensorDataset(*dataset[:train_size])
test_dataset = TensorDataset(*dataset[train_size:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# 初始化模型、损失函数和优化器
model = models.Normal_RNN(input_size, hidden_size, output_size, num_layers, model_type = model_type).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # print(inputs.shape)
        # print(targets.shape)

        # 训练时每次输入 sequence_length - 1 个数据，预测第 sequence_length 个数据
        output, hn = model(inputs, h0)   #inputs = [batch_size, sequence_length-1, features]
        h0 = hn.detach() # [layers, batch_size, hidden_size]
        #print(output.shape)
        #print(hn.shape)
        predictions = output[:, -1]

        #print(predictions.shape)
        #print(targets.shape)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward() # retain_graph=True
        max_norm = 2.0  # 设定梯度的最大范数，梯度更新限制越大
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # 使用clip_grad_norm_()函数控制梯度
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.7f}')

print("Training complete.")

# 模型评估
model.eval()
test_loss = 0.0
h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
with torch.no_grad():
    for inputs, targets in test_loader:
        #print(inputs.shape)
        #print(targets.shape)

        # 预测时每次输入 sequence_length - 1 个数据，预测第 sequence_length 个数据
        output, hn = model(inputs, h0)
        h0 = hn.detach()
        predictions = output[:, -1]

        loss = criterion(predictions, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.7f}')

## RNN
# Test Loss: 6.7714215 Ep=10, LR=1e-3 
# Test Loss: 0.0841337 Ep=10, LR=1e-4
# Test Loss: 0.0620537 Ep=10, LR=5e-5 | Epoch [10/10], Loss: 0.0000688
# Test Loss: 0.0555936 Ep=10, LR=1e-5 | Epoch [10/10], Loss: 0.0000089

## GRU
# Test Loss: 0.0495339 Ep=10, LR=1e-5 | Epoch [10/10], Loss: 0.0000048
# Test Loss: 0.0875661 Ep=20, LR=1e-5 | Epoch [20/20], Loss: 0.0000326

# Test Loss: 0.0745597 Ep=10, LR=1e-6 | Epoch [10/10], Loss: 0.0000036
# Test Loss: 0.0763733 Ep=20, LR=1e-6 | Epoch [20/20], Loss: 0.0004341
# Test Loss: 0.0602298 Ep=20, LR=1e-5 | Epoch [20/20], Loss: 0.0000151
# Test Loss: 0.1652189 Ep=20, LR=1e-5 | Epoch [20/20], Loss: 0.0000038 | input_size = 32 => 64
# Test Loss: 0.0413432 Ep=20, LR=1e-6 | Epoch [20/20], Loss: 0.0000610 | input_size = 32 => 64 hidden_size = 64 => 128
# Test Loss:  Ep=20, LR=1e-6 | Epoch [20/20], Loss:  | input_size = 32 hidden_size = 64 => 128
# Test Loss: 0.0897933 Ep=20, LR=1e-6 | Epoch [20/20], Loss: 0.0003652 | input_size = 32 => 64 hidden_size = 64 => 128 Layers = 3
