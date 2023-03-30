import torch
import csv
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from modle import MLP_model


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
data = []
df = pd.read_csv("rain_data_laster.csv", encoding="utf-8")
data_array = np.array(df)
# data_list = data_array.tolist()
# print(data_list)
x = data_array[:, 1:211]
y = data_array[:, -1].reshape(-1, 1)

# feature_range控制压缩数据范围，默认[0,1]
scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围

X = scaler.fit_transform(x)  # 将特征归一化到0,1之间
Y = scaler.fit_transform(y)  # 将标签归于化到0,1之间
# y = scaler.inverse_transform(Y) # 将数据恢复至归一化之前
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)

torch_dataset = torch.utils.data.TensorDataset(X, Y)
#随机划分数据集
torch.manual_seed(seed=1)
train, validation = torch.utils.data.random_split(
   torch_dataset, [5031, 2157])


#按照比例顺序划分数据集
# X = torch.tensor(X, dtype=torch.float32).to(device)
# Y = torch.tensor(Y, dtype=torch.float32).to(device)
# train_dataset_lenth = int(len(X)*0.7)
# validation_dataset_lenth = int(len(X)*0.3)
# test_dataset_lenth = len(X)-train_dataset_lenth-validation_dataset_lenth
# X_train = X[:train_dataset_lenth]
# Y_train = Y[:train_dataset_lenth]
# X_validation = X[train_dataset_lenth:train_dataset_lenth+validation_dataset_lenth]
# Y_validation = Y[train_dataset_lenth:train_dataset_lenth+validation_dataset_lenth]
# X_test = X[train_dataset_lenth+validation_dataset_lenth:]
# Y_test = Y[train_dataset_lenth+validation_dataset_lenth:]
#
# train = torch.utils.data.TensorDataset(X_train, Y_train)
# validation = torch.utils.data.TensorDataset(X_validation, Y_validation)
# test = torch.utils.data.TensorDataset(X_test, Y_test)
batch_size = 50
train_data = torch.utils.data.DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=False)
# val_data = torch.utils.data.DataLoader(  validation,
#                                          batch_size=batch_size,
#                                          shuffle=True)

net = MLP_model(n_feature = 210,
                n_output = 1,
                n_neuron1= 1024,
                n_neuron2= 8192,
                n_neuron3= 512,
                n_neuron4= 128
                )
net.to(device)
optimizer = optim.Adam(net.parameters(), lr = 2.8398814299996842e-05)  # 使用Adam算法更新参数
loss_function = torch.nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
epochs = 1000
save_path = './rain_radar_laster.pth'
train_steps = len(train_data)
val_num = len(validation)
best_acc = 0.0

for epoch in range(epochs):  # 整个数据集迭代次数
    net.train()  # 启动训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_data):
        logits = net.forward(data).to(device) # 前向计算结果（预测结果）
        loss = loss_function(logits, target)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传递过程
        optimizer.step()  # 优化权重与偏差矩阵
        running_loss += loss.item()

    net.eval()  # 启动测试模式
    acc = 0
    logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    target = []
    b = []
    with torch.no_grad():
        for data, targets in validation:  # 输出验证集的平均误差
            logits = net.forward(data).detach()
            targets = targets.detach()
            b.append((logits.item(), targets.item()))
            target.append(targets[0])
            logit.append(logits[0])
            if (logits - targets)**2 <=0.01:
                acc += 1
        # acc += torch.eq(logits, targets).sum().item()
        accuray = (acc / val_num) * 100
        average_loss = loss_function(torch.tensor(logit).to(device), torch.tensor(target).to(device))
    print("[epoch %d] train_loss: %.8f average_loss: %.8f accuray: %.3f" %
          (epoch+1, running_loss, average_loss, accuray ))

    if (epoch+1) % 50 == 0:
        torch.save(net.state_dict(), save_path)

print("logit:{}".format(logit))
print("target:{}".format(target))
print("b:{}".format(b))