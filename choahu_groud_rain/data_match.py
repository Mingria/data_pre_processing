from parse_radar_file import  diamond131
from parse_meteorological import parse_txt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm

radar_path = "D:\\PycharmProject\\dataset\\meteorological_data\\rain_dataset\\"
def read_radar_name():

    file_name = os.listdir(radar_path)
    file_name.sort()
    return file_name
# name = read_radar_name()
# print(name)

# radar1 = diamond131()
# radar1.ReadFile("D:\\PycharmProject\\dataset\\meteorological_data\\rain_dataset\\Z_OTHE_RADAMOSAIC_20190605164800.bin")
# real_value = radar1.ObserverValue
groud_data = parse_txt() #地面标签
name = read_radar_name() #雷达文件名称
##判断读取数据的值
# b = []
# i = 0
# for element in real_value.flat:
#      if element:
#          i+=1
#          b.append(element)
#
# print('b ',b)
# print('i ',i)
albedo_and_preciptation = []

for i, groud in enumerate(groud_data):
    rate = 0
    num = 0
    if groud[-2]:
        for path in name[(groud[-2]-1)*10 : groud[-2]*10]:
            path = os.path.join(radar_path, path)
            radar1 = diamond131()
            radar1.ReadFile(path)
            real_value = radar1.ObserverValue
            #匹配前一个小时的雷达数据
            assert groud[-2] - radar1.Hour == 1, "匹配错误"
            num += 1
            rate += real_value
        #取这一个小时的平均数据
        val_rate = rate/num
        albedo = val_rate[:, groud[0], groud[1]]
        albedo_list = list(albedo)
        albedo_list.append(groud[-1])
        albedo_and_preciptation.append(albedo_list)
    print("完成度{:.2f}%".format((i/len(groud_data))*100))

print(albedo_and_preciptation)
array = np.array(albedo_and_preciptation)
df = pd.DataFrame(array)
df.to_csv("rain_dara.csv")












# real_value_tensor = torch.tensor(real_value)
# x_start = radar1.StartLon
# y_start = radar1.StartLat
# x_reso = radar1.XReso
# y_reso = radar1.YReso
# x = [x_start]
# y = [y_start]
# for i in range(radar1.XNumGrids-1):
#     x.append(x_start + x_reso * (i+1))
# for j in range(radar1.YNumGrids-1):
#     y.append(y_start - y_reso * (j+1))
#
# print('x', x)
# print('y',y)
# r = real_value[20]
# x = x * 900
# y = y * 800
# x_np = np.array(x).reshape(900, 800)
# y_np = np.array(y).reshape(900, 800)







