import torch
import csv
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from modle import MLP_model


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
data = []
df = pd.read_csv("./rain_dara.csv",encoding="utf-8")
data_array = np.array(df)
# data_list = data_array.tolist()
# print(data_list)
x = data_array[:, 1:22]
y = data_array[:, -1].reshape(-1, 1)

# feature_range控制压缩数据范围，默认[0,1]
scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围

X = scaler.fit_transform(x)  # 将标签归一化到0,1之间
Y = scaler.fit_transform(y)  # 将特征归于化到0,1之间
# x = scaler.inverse_transform(X) # 将数据恢复至归一化之前


X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)
torch_dataset = torch.utils.data.TensorDataset(X, Y)
batch_size = 65

# 划分训练集测试集与验证集
torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
train, validation = torch.utils.data.random_split(
    torch_dataset,
    [5750, 1438],
)

# 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
train_data = torch.utils.data.DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=True)
# val_data = torch.utils.data.DataLoader(  validation,
#                                          batch_size=batch_size,
#                                          shuffle=True)
# vali_data = torch.utils.data.DataLoader()
net = MLP_model(n_feature = 21,
                n_output = 1,
                n_neuron1= 102,
                n_neuron2= 132
                )
net.to(device)
optimizer = optim.Adam(net.parameters(), lr = 2.8398814299996842e-05)  # 使用Adam算法更新参数
loss_function = torch.nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
epochs = 300
save_path = './rain_radar.pth'
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

    logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    target = []
    net.eval()  # 启动测试模式
    acc = 0
    with torch.no_grad():
        for data, targets in validation:  # 输出验证集的平均误差
            logits = net.forward(data).detach()
            targets = targets.detach()
            target.append(targets[0])
            logit.append(logits[0])
            if (logits - targets)*2 <=0.01:
                acc += 1
            # acc += torch.eq(logits, targets).sum().item()
        accuray = (acc / val_num) * 100
        average_loss = loss_function(torch.tensor(logit).to(device), torch.tensor(target).to(device))

    # val_accurate = acc / val_num
    # print('\nTrain Epoch:{} for the Average loss of VAL')
    print("[epoch %d] train_loss: %.8f average_loss: %.8f accuray: %.3f" %
          (epoch+1, running_loss / train_steps, average_loss, accuray ))

    if epoch % 50 == 0:

        torch.save(net.state_dict(), save_path)

print("logit:{}".format(logit))
print("target:{}".format(target))