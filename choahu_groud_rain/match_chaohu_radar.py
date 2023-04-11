import numpy as np
import os
import pandas as pd
import csv
import gc

import torch

from parse_chaohu import match_chaohu_radar
from data_match_laster import read_radar_name
from parse_radar_file import diamond131

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
def match_chaohu_radar_834_240_21():
    map, area = match_chaohu_radar()
    radar_name = read_radar_name(radar_path = "D:\\PycharmProject\\dataset\\meteorological_data\\rain_dataset\\")

    position = []
    for index_y, y in enumerate(map):
        for index_x, x in enumerate(y):
            if x == 255:
                position.append((index_y, index_x))
    # map_tensor = torch.tensor(map).to(device)
    # a = np.array(240,21,900,800)
    # b = a.reshape(-1,900,800)
    radar_data = []

    for i ,path_ in enumerate(radar_name):
        path = os.path.join("D:\\PycharmProject\\dataset\\meteorological_data\\rain_dataset\\", path_)
        radar1 = diamond131()
        radar1.ReadFile(path)
        real_value = radar1.ObserverValue
        radar_data.append(real_value)
        print("读取雷达数据：{:.2f}%".format((i/len(radar_name)*100)))
    print("雷达读取完成")
    radar_array = np.array(radar_data, dtype=np.float16).reshape(-1,900,800)

    chaohu_radar = []
    for i, pos in enumerate(position):
        chaohu_radar.append(radar_array[:, pos[0], pos[1]])
        print("匹配完成度:{:.2f}%".format((i/len(position))*100))

    array = np.array(chaohu_radar, dtype= np.float16).reshape(834,-1)
    df = pd.DataFrame(array)
    df.to_csv("chaohu_radar_834_240_21.csv")




