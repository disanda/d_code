import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import models

# 读取和预处理数据
file_path = 'mpi_saale_2021b.csv'
df = pd.read_csv(file_path)
print(df.columns)
data = df.drop(columns=['Date Time']) #去掉字符串特征

# 去掉其他degC特征
deg_columns = df.filter(like='degC').columns 
filtered_list = [item for item in deg_columns if item != 'T (degC)']
data = data.drop(columns=pd.Index(filtered_list))
#print(data.columns)

# 超参数
input_size = len(data.columns)
hidden_size = 64
output_size = 1
num_layers = 2
num_epochs = 30
learning_rate = 5e-5 #0.001
batch_size = 8
sequence_length = 8  # 输入9个特征，预测第10个特征
model_type ='RNN' # GRU

# 标准化特征
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
#print(data_scaled)

# 创建序列数据
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length-1])  # 前9个时间步的特征
    y.append(data_scaled[i+sequence_length-1, 1])  #  第10个时间步的第2个特征'T (degC)'作为目标

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
model = models.WeatherRNN(input_size, hidden_size, output_size, num_layers, model_type = model_type).to(device)
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

        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward() # retain_graph=True
        max_norm = 2.0  # 设定梯度的最大范数
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

# Test Loss: 0.0277881 Ep=20, LR=1e-4
# Test Loss: 0.0210380 Ep=30, LR=5e-5
# Test Loss: 0.0187615 Ep=21, LR=5e-5, hidden_size = 32 => 21
# Test Loss: 0.0160574 Ep=30
# Test Loss: 0.0087513 Ep=25, hidden_size = 32, RNN => GRU, Epoch [25/25], Loss: 0.0004827
# Test Loss: 0.0074449 Ep=30, hidden_size = 64, RNN => GRU, Epoch [30/30], Loss: 0.0004222



