
import supprting_function
import pandas as pd

import pathlib
import random
import antoencoder
import os

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
        accurcy = main.rf(file_g, file_original, lensc=SeqLength, ind=ind)
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

def RNN_core(file_0,file_1,SeqLength,repeated_num,j):
    PATH_generated_csv = []
    file_csv_rnns =  r'D:\Desktop\PaperCode_2023\ExperimentData\{}\stable_test_csv'
    pathlib.Path(file_csv_rnns).mkdir(parents=True, exist_ok=True)

    if SeqLength == 200:
        ind = '4'
    else:
        ind = '3'

    All_RNNS = pd.DataFrame(columns=['path', 'accurcy'])

    file_g = file_0
    file_original = file_1

    with open(file_g, 'r') as f1:
        lines = f1.readlines()

    supprting_function.csv_shengc(file_g, file_csv_rnns, file_original, len(lines), Seqlength=SeqLength)

    for root, dirs, files in os.walk(file_csv_rnns):
        for file in files:
            PATH_generated_csv.append(os.path.join(root, file))

    for file_g_ in PATH_generated_csv:
        t = file_g_.rfind('\\')
        name = file_g_[t + 1: len(file_g_) - 4]
        for j in range(repeated_num):
            accurcy, rnns_reaclls, rnns_pres, rnns_f1s = antoencoder.RNNs_Classifier(file_g_)
            All_RNNS = All_RNNS.append([{'path': name, 'accurcy': accurcy}], ignore_index=True)

        accurcy_average, accurcy_up, accurcy_down, accurcy_std = supprting_function.data_analysis(All_RNNS,
                                                                                                  type='accurcy')

        All_RNNS = All_RNNS.append([{'path': 'ave', 'accurcy': accurcy_average}], ignore_index=True)
        All_RNNS = All_RNNS.append([{'path': 'ACC_STD', 'accurcy': accurcy_std}], ignore_index=True)
        All_RNNS = All_RNNS.append([{'path': 'Pos ERR', 'accurcy': accurcy_up}], ignore_index=True)
        All_RNNS = All_RNNS.append([{'path': 'Neg ERR', 'accurcy': accurcy_down}], ignore_index=True)

        print(All_RNNS)

    All_RNNS.to_csv(r'D:\Desktop\PaperCode_2023\ExperimentData\{}\{}_RNNs_{}.csv'.format(i, i,j))


if __name__ == '__main__':
    for i in ['611', '572', '43']:
        file_original = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\OriginalData\{}.txt'.format(i, i)

        PATH_generated_path = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\baseline_xiaorong'.format(i)

        file_dirs = r'D:\Desktop\PaperCode_2023\ExperimentData\{}\stable_test'.format(i)

        repeated_j = 3
        for j in range(repeated_j):
            file_0, file_1 = split_data(file_original, file_dirs)
            RNN_core(file_0, file_1, SeqLength=200, repeated_num=2, j=j)








