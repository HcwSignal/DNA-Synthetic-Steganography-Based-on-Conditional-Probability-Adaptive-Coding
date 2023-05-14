import os
import numpy
import random
import copy
import numpy as np
import cg_tm_kl
import pandas as pd

if __name__ == '__main__':
    modes = ['572']
    sl = '4'
    for mode in modes:
        result = pd.DataFrame(columns=[])

        if mode == '611':
            length_ = 3000

        elif mode == '660':
            length_ = 3300

        elif mode == '43':
            length_ = 6600

        elif mode == '572':
            length_ = 2700


        path = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\read_{}.txt'.format(mode,sl)

        if int(sl) % 2 == 0:
            lensc = 200
            dividenum = int(sl)
        else:
            lensc = 198
            dividenum = int(sl)

        linesori = cg_tm_kl.txt_process_sc_duo(path, len_sc=lensc, beg_sc=0, end_sc=length_, PADDING=False, flex=0,
                                                devide_num=dividenum)

        dirs_sc = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\baselines'.format(mode)
        PATHSC = []
        for root,dirs,files in os.walk(dirs_sc):
            for file in files:
                PATHSC.append(os.path.join(root,file))

        for pathsc in PATHSC:
            linessc = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=lensc, beg_sc=0, end_sc=length_, PADDING=False, flex=0,devide_num=dividenum)

            CGd = cg_tm_kl.CG_b(linesori, linessc, len_sc=lensc)
            Tmd = cg_tm_kl.Tmb(linesori, linessc, len_sc=lensc)
            klds = cg_tm_kl.KLDoubleStrand(linessc, linesori)

            filename = pathsc[ pathsc.rfind('\\') +1 : -4]
            # print('Tmd:', Tmd)
            # print('CGD:', CGd)
            # print('KLDS', klds)
            print(1)

            result = result.append({'path':filename,'GCd':CGd,'Tmd':Tmd,'KLp':klds},ignore_index=True)

        writefile = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}_baselines_gcb_tmb_klp.csv'.format(mode)
        result.to_csv(writefile)











