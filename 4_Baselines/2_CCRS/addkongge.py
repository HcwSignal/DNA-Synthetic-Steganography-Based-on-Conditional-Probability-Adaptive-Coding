with open(r"D:\Destop\non_coding\class.txt","r") as f1:
    lines = f1.readlines()
f2 = open(r"D:\Destop\non_coding\nc_class_kongge.txt",'w')
num = 0
for out in lines:
    three_base = []
    for j in range(0, len(out) - 2, 3):
        shuchu = out[j] + out[j + 1] + out[j + 2]
        three_base.append(shuchu)
    out = ' '.join(three_base) + ' '
    f2.write(out)
    f2.write("\n")
    num  += 1
    if num == 1452:
        break

f2.close()