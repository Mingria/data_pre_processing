import numpy as np
from PIL import Image
import math


def match_chaohu_radar():
    chaohu = parse_bln()
    # 建立与雷达图相同的特征地图，用来匹配巢湖地区
    map = np.array([[0 for i in range(800)] for j in range(900)])
    #四舍五入定边界
    # for position in chaohu:
    #     if (float(position[0])-113.0)/0.01 - (float(position[0])-113.0)//0.01 < 0.5:
    #         x = int((float(position[0])-113.0)//0.01)
    #     else: x = int((float(position[0])-113.0)//0.01 + 1)
    #     if (36.0-float(position[1]))//0.01 - (36.0-float(position[1]))//0.01 < 0.5:
    #         y = int((36.0-float(position[1]))//0.01)
    #     else: y = int((36.0-float(position[1]))//0.01 + 1)
    #
    #     map[y, x] = 255
    #取巢湖边界
    for position in chaohu:
        x = int((float(position[0])-113.0)//0.01)
        y = int((36.0-float(position[1]))//0.01)
        map[y, x] = 255
    # odd_line = []  #奇数行
    # even_line = [] #偶数行
    pos = []
    #获取匹配雷达图后的逐个点坐标进而确定地区
    for index_y, value_y in enumerate(map):
        row = []
        if sum(value_y):
            for index_x, value_x in enumerate(value_y):
                if value_x:
                    row.append(index_x)
            pos.append((index_y,row))
        # even_line_row = []  # 偶数行列索引
        # odd_line_row = []  # 奇数行列索引
        #偶数行
        # if sum(value_y) and sum(value_y) % 510 == 0:
        #     for index_x, value_x in enumerate(value_y):
        #         if value_x:
        #             even_line_row.append(index_x)
        #
        #     even_line.append((index_y,even_line_row))
        #
        #
        #     # even_index.append((index_y,sum(value_y)/255))
        # #奇数行
        # if  sum(value_y) and sum(value_y) % 510 == 255:
        #     for x_index, x_value in enumerate(value_y):
        #         if x_value:
        #             odd_line_row.append(x_index)
        #     odd_line.append((index_y, odd_line_row))

    map[428, 433:439] = 255
    map[429, 431:439] = 255
    map[430, 430:444] = 255
    map[431, 429:446] = 255
    map[432, 429:446] = 255
    map[433, 429:446] = 255
    map[433, 465:468] = 255
    map[434, 429:445] = 255
    map[434, 463:473] = 255
    map[435, 428:445] = 255
    map[435, 462:474] = 255
    map[436, 428:445] = 255
    map[436, 462:475] = 255
    map[437, 429:445] = 255
    map[437, 461:479] = 255
    map[438, 429:446] = 255
    map[438, 459:480] = 255
    map[439, 429:446] = 255
    map[439, 458:483] = 255
    map[440, 429:446] = 255
    map[440, 457:483] = 255
    map[441, 429:455] = 255
    map[441, 457:482] = 255
    map[442, 430:482] = 255
    map[443, 431:476] = 255
    map[444, 433:476] = 255
    map[445, 432:476] = 255
    map[446, 439:475] = 255
    map[447, 438:472] = 255
    map[448, 442:472] = 255
    map[449, 442:471] = 255
    map[450, 443:469] = 255
    map[451, 444:468] = 255
    map[452, 444:468] = 255
    map[453, 445:467] = 255
    map[454, 448:465] = 255
    map[455, 449:463] = 255
    map[456, 451:462] = 255
    map[457, 454:456] = 255
    line_left = [] #左边界索引（经度）
    line_y = []    #右边界索引（经度）
    line_right = []#行索引（纬度）
    for id_y ,y in enumerate(map):
        for id_x ,x in enumerate(y):
            if x and y[id_x-1] == 0 and y[id_x+1] == 255:
                line_left.append(id_x)
                line_y.append(id_y)
            if x and y[id_x+1] == 0 and y[id_x-1] == 255:
                line_right.append(id_x)
        line = np.array(line_right) - np.array(line_left)
    length = []  #纬度下对应的左右边界距离
    height = []  #经度下对应的上下两行的距离
    for i, left in enumerate(line_left):
        #计算横向距离（同一纬度下经度的距离）
        distance = getDistance(latA=36-line_y[i]*0.01, lonA=left*0.01+113.0 ,latB=36-line_y[i]*0.01, lonB=line_right[i]*0.01+113)
        length.append(distance)
    for j, high in enumerate(line_y):
        #计算纵向距离（同一经度下纬度距离）
        high = getDistance(latA=36-high*0.01,lonA=line_left[j]*0.01+113, latB=36-line_y[j]*0.01-0.01, lonB=line_right[j]*0.01+113)
        height.append(high)
    #巢湖面积
    area = sum(np.array(distance) * np.array(height))
    return map,area



def parse_bln( path="D:\\PycharmProject\\dataset\\meteorological_data\\chaohu.bln"):
    #解析巢湖经纬度
    file_path = path
    chaohu = []
    with open(file_path, 'r' ,encoding='utf-8') as f:
        for line in f.readlines():
            file = line.split()
            chaohu.append(file)
    #将无关信息删除，如 136   1
    for index, i in enumerate(chaohu):
        if float(i[-1]) -1.0 == 0.0:
            del chaohu[index]
    return chaohu


# 计算距离
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # 赤道半径
    rb = 6356755  # 极半径
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance / 1000, 4)
    return distance

if __name__ == '__main__':

    val = match_chaohu_radar()
    # im = Image.fromarray(val)
    # im.save("banben_1.png")
    # im.show()

    # print(val.sum()/255)


