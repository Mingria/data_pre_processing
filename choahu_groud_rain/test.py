import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from modle import MLP_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

chaohu = pd.read_csv("chaohu_radar_834_240_21.csv", encoding="utf-8")
scaler = MinMaxScaler(feature_range=[0, 1])
chaohu_radar_array = np.array(chaohu)[:, 1:].reshape(834,24,210).transpose(1,0,2)
# chaohu_scaler = scaler.fit_transform(chaohu_radar_array)
# a = torch.tensor([-1,2,3,4,5])
# a[a<0]= 0
# print(sum(a))

# chaohu_tensor = torch.tensor(chaohu_scaler).to(device)


modle = MLP_model(n_feature = 210,
                  n_output = 1,
                  n_neuron1= 1024,
                  n_neuron2= 8192,
                  n_neuron3= 512,
                  n_neuron4= 128).to(device)
weights_path = "./rain_radar_laster.pth"
assert  os.path.exists(weights_path), "file: '{}' dose no exist".format(weights_path)
modle.load_state_dict(torch.load(weights_path, map_location= device))
modle.eval()  # 启动测试模式
with torch.no_grad():
    f = open('chaohu_rain_24hour', 'w', encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["2019/6/5-6/6日巢湖每小时降雨量"])
    for i in range(24):
        chaohu = chaohu_radar_array[i]
        chaohu_scaler = scaler.fit_transform(chaohu)
        chaohu_tensor = torch.tensor(chaohu_scaler, dtype= torch.float32).to(device)
        output = modle(chaohu_tensor)
        output[output<0] = 0
        sum_out = sum(output)/834
        csv_writer.writerow(['第{}小时降水量'.format(i+1),sum_out.item()])
        print("第{}小时降水量".format(i+1),sum_out.item())
        # print(output)
    f.close()

