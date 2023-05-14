import numpy as np
import pandas as pd
import cg_tm_kl
import pathlib
def data_analysis(temp,type):
    accurcy_average = np.mean(pd.to_numeric(temp[type]).tolist())

    accurcy_up = max(pd.to_numeric(temp[type]).tolist()) - accurcy_average

    accurcy_down = accurcy_average - min(pd.to_numeric(temp[type]).tolist())

    accurcy_std = np.std((pd.to_numeric(temp[type]).tolist()))

    return accurcy_average,accurcy_up,accurcy_down,accurcy_std

def csv_shengc(pathsc,file_csv,ori, sample_num,Seqlength):
    dirs_csv = file_csv
    pathori = ori

    raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc= Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

    pathlib.Path(r'D:\Desktop\seqs\Ho Bae\tt').mkdir(parents=True,exist_ok=True)

    pathwrite_sc = r'D:\Desktop\seqs\Ho Bae\tt\1.txt'

    with open(pathwrite_sc, 'w') as f1:
        for line in raw_pos:
            f1.write(line)
            f1.write('\n')

    raw_neg = cg_tm_kl.txt_process_sc_duo(pathori, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

    pathwrite_ori = r'D:\Desktop\seqs\Ho Bae\tt\2.txt'
    with open(pathwrite_ori, 'w') as f2:
        for line in raw_neg:
            f2.write(line)
            f2.write('\n')

    pos = pd.read_csv(pathwrite_sc, header=None)
    neg = pd.read_csv(pathwrite_ori, header=None)

    x = np.concatenate((pos, neg))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    save = pd.DataFrame(columns=['text', 'label'], index=range(len(x)))

    save['text'] = x
    save['label'] = y

    # t = pathsc[pathsc.find('traditionnal method') +len('traditionnal method') + 1  : len(pathsc)-4]

    t = pathsc[pathsc.rfind('\\'): len(pathsc) - 4]
    file_ = dirs_csv + t + '.csv'

    save.to_csv(file_)
