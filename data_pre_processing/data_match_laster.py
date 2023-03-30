from parse_radar_file import  diamond131
from parse_txt import parse_txt
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
groud_data = parse_txt()
name = read_radar_name()
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
    if groud[-2]:
        ten = []
        for path in name[(groud[-2]-1)*10 : groud[-2]*10]:
            path = os.path.join(radar_path, path)
            radar1 = diamond131()
            radar1.ReadFile(path)
            real_value = radar1.ObserverValue
            assert groud[-2] - radar1.Hour == 1, "匹配错误"
            ten.append(real_value)
        concat = np.array(ten).reshape(-1, 900, 800)
        albedo = concat[:, groud[0], groud[1]]
        albedo_list = list(albedo)
        albedo_list.append(groud[-1])
        albedo_and_preciptation.append(albedo_list)


    print("完成度{:.2f}%".format((i/len(groud_data))*100))

print(albedo_and_preciptation)
array = np.array(albedo_and_preciptation)
df = pd.DataFrame(array)
df.to_csv("rain_data_laster.csv")








