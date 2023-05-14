import rf_functinn
import supprting_function
import pandas as pd
import os
import pathlib
import random
def split_data(path_original,file_dirs):
    pathlib.Path(file_dirs).mkdir(parents=True,exist_ok=True)
    with open(path_original,'r') as f1:
        lines = f1.readlines()

    #是否对数据进行随机打乱
    random.shuffle(lines)

    file_write_0 = file_dirs + r'\0.txt'.format(i)
    file_write_1 = file_dirs + r'\1.txt'.format(i)

    lines_ = lines[ : len(lines) - len(lines)%2 ] #若有不能整除2的多余数据行，清除
    lines_write_0 = lines_[  :int(len(lines_) / 2 )  ]
    lines_write_1 = lines_[int(len(lines_) / 2 ) :     ]

    with open(file_write_0,'w') as f_0:
        for line in lines_write_0:
            f_0.write(line)

    with open(file_write_1, 'w') as f_1:
        for line in lines_write_1:
            f_1.write(line)

    return file_write_0,file_write_1

def RF_core(file_0,file_1,SeqLength,repeated_num,j):
    #SeqLength,repeated_num = int,str,int
    if SeqLength == 200:
        ind = '4'
    else:
        ind = '3'
    All_RF = pd.DataFrame(columns=['path', 'accurcy'])

    file_g = file_0
    file_original = file_1

    for j in range(repeated_num):
        t = file_g.rfind('\\')
        name = file_g[t + 1: len(file_g) - 4] + r'_{}'.format(j)
        accurcy = rf_functinn.rf(file_g, file_original, lensc=SeqLength, ind=ind)
        All_RF = All_RF.append([{'path': name, 'accurcy': accurcy}], ignore_index=True)
        print("finish:{}/{}".format(j,repeated_num))


    accurcy_average, accurcy_up, accurcy_down, accurcy_std = supprting_function.data_analysis(All_RF,
                                                                                              type='accurcy')

    All_RF = All_RF.append([{'path': 'ave', 'accurcy': accurcy_average}], ignore_index=True)
    All_RF = All_RF.append([{'path': 'ACC_STD', 'accurcy': accurcy_std}], ignore_index=True)
    All_RF = All_RF.append([{'path': 'Pos ERR', 'accurcy': accurcy_up}], ignore_index=True)
    All_RF = All_RF.append([{'path': 'Neg ERR', 'accurcy': accurcy_down}], ignore_index=True)

    print(All_RF)

    All_RF.to_csv(r'D:\Desktop\PaperCode_2023\ExperimentData\{}\{}_RF_{}.csv'.format(i, i,j))

for i in ['611','572','43']:
    file_original = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\OriginalData\{}.txt'.format(i, i)

    PATH_generated_path = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\baseline_xiaorong'.format(i)

    file_dirs = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\stable_test'.format(i)

    repeated_j = 3
    for j in range(repeated_j):
        file_0,file_1 = split_data(file_original,file_dirs)
        RF_core(file_0,file_1,SeqLength=200,repeated_num=2,j=j)






