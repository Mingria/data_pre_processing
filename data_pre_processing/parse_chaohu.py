import numpy as np

def parse_bln( path="D:\\PycharmProject\\dataset\\meteorological_data\\chaohu.bln"):

    file_path = path
    chaohu = []
    b = []
    value = []
    with open(file_path, 'r' ,encoding='utf-8') as f:
        for line in f.readlines():
            file = line.split()
            chaohu.append(file)


    print(a1)
    for index, i in enumerate(a):
        if float(i[-1]) -1.0 == 0.0:
            del chaohu[index]
    map = [[0 for i in range(800)] for j in range(900)]
    for label in chaohu:
        if (float(label[0])-113.0)/0.01 - (float(label[0])-113.0)//0.01 < 0.5:
            x = (float(label[0])-113.0)//0.01
        else: x = (float(label[0])-113.0)//0.01 + 1
        if (float(label[1])-113.0)//0.01 - (float(label[1])-113.0)//0.01 < 0.5:
            y = (float(label[0])-113.0)//0.01

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
val = parse_bln()
print(val)








