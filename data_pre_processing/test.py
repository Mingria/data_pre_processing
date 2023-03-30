import matplotlib.pyplot as plt
import numpy as np

prediction = []
test_y = []
net.eval()  # 启动测试模式
for test_x, test_ys in test:
    predictions = net(test_x)
    predictions = predictions.detach().numpy()
    prediction.append(predictions[0])
    test_ys.detach().numpy()
    test_y.append(test_ys[0])
prediction = scaler.inverse_transform(np.array(prediction).reshape(
    -1, 1))  # 将数据恢复至归一化之前
test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))
# 均方误差计算
test_loss = criteon(torch.tensor(prediction, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
print('测试集均方误差：', test_loss.detach().numpy())

# 可视化
plt.figure()
plt.scatter(test_y, prediction, color='red')
plt.plot([0, 52], [0, 52], color='black', linestyle='-')
plt.xlim([-0.05, 52])
plt.ylim([-0.05, 52])
plt.xlabel('true')
plt.ylabel('prediction')
plt.title('true vs prection')
plt.show()
