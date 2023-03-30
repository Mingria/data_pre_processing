import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = []
df = pd.read_csv("rain_data_laster.csv", encoding="utf-8")
data_array = np.array(df)
x = data_array[:, 1:211]
y = data_array[:, -1].reshape(-1, 1)

# feature_range控制压缩数据范围，默认[0,1]
scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围

X = scaler.fit_transform(x)  # 将特征归一化到0,1之间
# Y = scaler.fit_transform(y)  # 将标签归于化到0,1之间
# # y = scaler.inverse_transform(Y) # 将数据恢复至归一化之前

X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(y, dtype=torch.float32).to(device)
torch_dataset = torch.utils.data.TensorDataset(X, Y)
batch_size = 6


torch.manual_seed(seed= 1)
train, validation = torch.utils.data.random_split(
    torch_dataset,
    [5031, 2157]
)

class Net(torch.nn.Module):
    '''搭建神经网络'''

    def __init__(
            self, n_feature, n_output, n_neuron1, n_neuron2,
            n_layer):
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_neuron1 = n_neuron1
        self.n_neuron2 = n_neuron2
        self.n_layer = n_layer
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature,
                                           self.n_neuron1)
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2)
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2)
        self.predict = torch.nn.Linear(self.n_neuron2, self.n_output)

    def forward(self, x):

        out = self.input_layer(x)
        out = torch.relu(out)
        out = self.hidden1(out)
        out = torch.relu(out)
        for _ in range(self.n_layer):
            out = self.hidden2(out)
            out = torch.relu(out)
        out = self.predict(out)
        return out


def structure_initialization(parameters):
    '''实例化神经网络'''
    n_layer = parameters.get('n_layer', 2)
    n_neuron1 = parameters.get('n_neuron1', 140)
    n_neuron2 = parameters.get('n_neuron2', 140)
    learning_rate = parameters.get('learning_rate', 0.0001)
    net = Net(n_feature=210,
              n_output=1,
              n_layer=n_layer,
              n_neuron1=n_neuron1,
              n_neuron2=n_neuron2)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 learning_rate)
    criteon = torch.nn.MSELoss()
    return net, optimizer, criteon


def train_evaluate(parameterization):
    '''此函数返回模型误差作为贝叶斯优化依据'''
    net, optimizer, criteon = structure_initialization(parameterization)
    batch_size = parameterization.get('batch_sizes', 6)
    epochs = parameterization.get('epochs', 100)

    train_data = torch.utils.data.DataLoader(train,
                                             batch_size=batch_size,
                                             shuffle=False)
    net.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_data):
            logits = net.forward(data)
            loss = criteon(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    logit = []
    target = []
    net.eval()
    with torch.no_grad():
        for data, targets in validation:
            logits = net.forward(data).detach()
            targets = targets.detach()
            target.append(targets[0])
            logit.append(logits[0])
        average_loss = criteon(torch.tensor(logit), torch.tensor(target))
    return float(average_loss)


from ax.service.managed_loop import optimize  # 使用贝叶斯优化超参数,可以使用pip install ax-platform命令安装，贝叶斯优化具体介绍见https://ax.dev/docs/bayesopt.html


def bayesian_optimization():
    best_parameters, values, experiment, model = optimize(
        parameters=[{
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-6, 0.1],
            "log_scale": True
        }, {
            "name": "n_layer",
            "type": "range",
            "bounds": [0, 4]
        }, {
            "name": "n_neuron1",
            "type": "range",
            "bounds": [40, 1000]
        }, {
            "name": "n_neuron2",
            "type": "range",
            "bounds": [40, 1000]
        }, {
            "name": "batch_sizes",
            "type": "range",
            "bounds": [6, 100]
        }, {
            "name": "epochs",
            "type": "range",
            "bounds": [50, 500]
        }],
        evaluation_function=train_evaluate,
        objective_name='MSE LOSS',
        total_trials=200,
        minimize=True)
    return best_parameters


best = bayesian_optimization()  # 返回最优的结构
print(best)