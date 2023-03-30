import torch.optim as optim
import torch

class MLP_model(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_neuron1, n_neuron2,n_neuron3,n_neuron4
                 ):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_neuron1 = n_neuron1
        self.n_neuron2 = n_neuron2
        self.n_neuron3 = n_neuron3
        self.n_neuron4 = n_neuron4

        super(MLP_model, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, self.n_neuron1)  # 输入层
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2)  # 1类隐藏层
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2)  # 2类隐藏
        self.hidden3 = torch.nn.Linear(self.n_neuron2, self.n_neuron3)
        self.hidden4 = torch.nn.Linear(self.n_neuron3, self.n_neuron4)
        self.predict = torch.nn.Linear(self.n_neuron4, self.n_output)  # 输出层
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.input_layer(x)
        out = torch.relu(out)  # 使用relu函数非线性激活
        out = self.dropout(out)
        out = self.hidden1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.hidden4(out)
        out = torch.relu(out)
        out = self.predict(out)  # 除去feature_number与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out


