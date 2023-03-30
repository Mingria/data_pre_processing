import numpy as np

def parse_txt( path="D:\\PycharmProject\\dataset\\meteorological_data\\meteorological_station.txt"):

    file_path = path
    a = []
    b = []
    value = []
    with open(file_path, 'r' ,encoding='utf-8') as f:
        for line in f.readlines():
            file = line.split()
            a.append(file)
    a.pop(0)
    for i in a:
        j = i[-7:]
        if j[-1] != '999999' and j[-1] != '999990':
            value.append(j)


    # for _ in value:
    #     for p in _:
    #         b.append(float(p))
    for j in value:
        index_x = int((float(j[1])-113.0)/0.01)
        index_y = int((36.0-float(j[0]))/0.01)
        if j[-3] == '6':
            time_ = 24
        else:time_ = int(j[-2])
        precipitation = float(j[-1])

        b.append([index_y, index_x, time_, precipitation])



    # vaule_np =  np.array(b).reshape(-1, 7)

    # print(vaule_np.shape)

    # print(np.max(vaule_np,axis=0))



    return b
# val = parse_txt()
# print(val)
#







