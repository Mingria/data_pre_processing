import numpy as np
from PIL import Image



def parse_bln( path="D:\\PycharmProject\\dataset\\meteorological_data\\chaohu.bln"):

    file_path = path
    chaohu = []
    with open(file_path, 'r' ,encoding='utf-8') as f:
        for line in f.readlines():
            file = line.split()
            chaohu.append(file)

    for index, i in enumerate(chaohu):
        if float(i[-1]) -1.0 == 0.0:
            del chaohu[index]
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
    #取边界
    for position in chaohu:
        x = int((float(position[0])-113.0)//0.01)
        y = int((36.0-float(position[1]))//0.01)
        map[y, x] = 255
    odd_number = []
    even_number = []
    even_index = []
    odd_index = []
    pos = []

    for index_y, value_y in enumerate(map):
        x_pos = []
        y_pos = []
        # for index_x, value_x in enumerate(value_y):
        #     if value_x:
        #         pos.append(index_x)
        if sum(value_y) and sum(value_y) % 510 == 0:
            for index_x, value_x in enumerate(value_y):
                if value_x:
                    x_pos.append(index_x)


            number = int(len(x_pos)/2)
            j = 0
            for i in range(number):
                if j < number:
                    map[index_y, x_pos[j]:x_pos[j+1]+1] = 255
                    j += 1

            even_index.append((index_y,sum(value_y)/255))
        if  sum(value_y) and sum(value_y) % 510 == 255:

            odd_index.append((index_y,sum(value_y)/255))




    # print("even_number: %d, odd_number: %d" % (even_number,odd_number))
    print("even_index:{}".format(even_index))
    print("odd_index:{}".format(odd_index))
    print("position:{}".format(pos))




    return map
val = parse_bln()
im = Image.fromarray(val)
# im.save("banben_1.png")
im.show()

print(val.sum()/255)






