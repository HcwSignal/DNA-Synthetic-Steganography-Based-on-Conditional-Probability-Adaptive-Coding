import os
import numpy as np
txt_file = open('D:\TextClassification\data\sc_10000.txt', 'r+')
des_txt_file = open('data/data_half/sc_two_5k_Per100.txt','w+')
'''
#为每段话加空格
lines = txt_file.readlines()

for line in lines:
    result = ''
    temp = line.strip()
    #print(temp)
    temp1 = temp.split()
    #print(temp1)
    temp2 = ''.join(temp1)
    #print(temp2)
    line_length = len(temp2)
    #for i in range(0, line_length):
    for i in range(0, line_length,2):
        result = result + temp2[i] + temp2[i + 1] + ' '
        #result = result + temp2[i] + ' '
        #continue
        # print('temp[i]', temp[i])
        # print(' type temp[i]', type(temp[i]))
        #result = result + temp[i] + temp[i+1]+' '
        #print('result', result)
    des_txt_file.write(result + '\n')
txt_file.close()
des_txt_file.close()
'''
lines = txt_file.readlines()
result = ''
for line in lines:
    temp = line.strip().split()
    result += ''.join(temp)

len_re = len(result)
Result = np.array(list(result)).reshape((-1,200))

for Re in Result:
    out = ''
    for i in range(0,len(Re),2):
        out += Re[i] + Re[i+1] + ' '

    des_txt_file.writelines(out + '\n')

    #des_txt_file.write(result + '\n')

txt_file.close()
des_txt_file.close()
