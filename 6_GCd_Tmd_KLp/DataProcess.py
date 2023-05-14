#功能：根据计算结果，自动计算每一个文件的指标平均值、正偏差、负偏差、标准差
#并在分类结果文件目录下自动生成所对应的四个文件

import csv
import pandas as pd
import math
import numpy as np
def main(mode):
    path = r'D:\Destop\论文代码\PaperCode\GCd_Tmd_KLp\{}.csv'.format(mode)
    file = pd.read_csv(path)
    Keys = ['GCd', 'Tmd', 'KLp']
    #index_ = ['2', '3', '4', '5', '6', '8', '10', '20', '33', '40', '50', '66']
    index_ = ['adg', 'DOUBLESUB', 'TLSM', 'Passing', 'GCBS']

    pd_writefile_     = pd.DataFrame(columns=[])
    pd_writefile_Std_ = pd.DataFrame(columns=[])
    pd_writefile_Pos_ = pd.DataFrame(columns=[])
    pd_writefile_Neg_ = pd.DataFrame(columns=[])

    for key in Keys:
        pd_writefile     = pd.DataFrame(columns=[])
        pd_writefile_Std = pd.DataFrame(columns=[])
        pd_writefile_Pos = pd.DataFrame(columns=[])
        pd_writefile_Neg = pd.DataFrame(columns=[])
        all = {}
        for index, row in file.iterrows():
            all[row['path'] + '-' + key] = row[key]

        for ind_ in index_:
            Value = []
            for element in all.items():
                if ind_ in element[0]:
                    Value.append(element[1])

            ValueMean = np.mean(np.array(Value))
            ValueMax  = max(Value)
            ValueMin  = min(Value)
            PosDeviation = ValueMax - ValueMean
            NegDeviation = ValueMean - ValueMin
            ValueStd  = np.std(np.array(Value))

            pd_writefile     = pd_writefile.append({key:ValueMean},ignore_index=True)
            pd_writefile_Std = pd_writefile_Std.append({key:ValueStd},ignore_index=True)
            pd_writefile_Pos = pd_writefile_Pos.append({ key: PosDeviation}, ignore_index=True)
            pd_writefile_Neg = pd_writefile_Neg.append({ key: NegDeviation}, ignore_index=True)

        pd_writefile_ = pd.concat([pd_writefile_,pd_writefile],axis=1)
        pd_writefile_Std_ = pd.concat([pd_writefile_Std_, pd_writefile_Std], axis=1)
        pd_writefile_Pos_ = pd.concat([pd_writefile_Pos_, pd_writefile_Pos], axis=1)
        pd_writefile_Neg_ = pd.concat([pd_writefile_Neg_, pd_writefile_Neg], axis=1)

    pd_writefile_.insert(0,'path',index_)
    pd_writefile_Std_.insert(0, 'path', index_)
    pd_writefile_Pos_.insert(0, 'path', index_)
    pd_writefile_Neg_.insert(0, 'path', index_)

    pd_writefile_.to_csv(path[ : -4] + "-avearge.csv")
    pd_writefile_Std_.to_csv(path[ : -4] + "-STD.csv")
    pd_writefile_Pos_.to_csv(path[ : -4] + "-PosDeviation.csv")
    pd_writefile_Neg_.to_csv(path[ : -4] + "-NegDeviation.csv")
    print(1)

if __name__ == '__main__':
    modes = ['888','847','660']
    for mode in modes:
        main(mode)