import numpy
import random
import copy
import math
import numpy as np
#import KL
import cg_tm_kl
import KL1
ALL = ['A','T','C','G']

def random_sub(k,rate):
    k = list(''.join(k.strip().split(' ')))
    index = random.sample(range(0, len(k)), int(math.ceil(rate * len(k))))
    for i in index:
        t = k[i]
        ALL.remove(k[i])
        temp = ALL[random.randint(0,2)]
        k[i] = temp
        ALL.append(t)

    line = ''.join(k)
    return line

def DOUBLESUB(linglist):
    linglist = linglist.strip().split(' ')
    linglist = ''.join(linglist)
    subindex = []
    i = 0
    while  i < len(linglist):
        try:
            if linglist[i] == linglist[i+1]:
                subindex.append(i+1)
                i += 2
            else:
                i += 1
        except:
            break

    linglist = list(linglist)
    for index in subindex:
        linglist[index] = ALL[random.randint(0,3)]

    linglist = ''.join(linglist)
    return linglist


if __name__ == '__main__':
    OnePercent = []
    TenPercent = []
    FivePercent = []
    TwentyPercnt = []

    LSBOUT = []
    jy = []
    mode = '660'
    path = r'D:\Destop\file\科研相关\论文代码\ExperimentData\43\base_extaction\read_4.txt' # reference sequence

    if mode == '611':
        length_ = 3000
        devide = 4
    elif mode == '660':
        length_ = 3300
        devide = 3
    elif mode == '43':
        length_ = 6600
        devide = 4
    elif mode == '572':
        length_ = 2700
        devide = 4


    split_suq = cg_tm_kl.txt_process_sc_duo(path,len_sc=200,beg_sc=0,end_sc=length_,PADDING=False,flex=0,devide_num = devide)


    for i in range(0,3):
        GCBSOUT = []
        TLSMOUT = []
        DOUBLESUNOUT = []
        for line in split_suq:
            temp = ''.join(line.strip().split(' '))
            line_DOU = DOUBLESUB(line)
            line_GCBS = random_sub(line,
                                   0.50)  # The pseudo-sequence of information encoded by GCBS method is equal in length to the original sequence,
            # so it can be considered as random substitution at 50% modification rate
            line_TLSM = random_sub(line,
                                   0.82)  # The BPN of TLSM is 1.64, hence it can be considered as random substitution at 82% modification rate

            line_1_percent_randomsub = random_sub(line, 0.01) #non-specialized methods

            line_5_percent_randomsub = random_sub(line,0.05) #non-specialized methods

            line_10_percent_randomsub = random_sub(line, 0.10) #non-specialized methods

            line_20_percent_randomsub = random_sub(line, 0.20) #non-specialized methods

            OnePercent.append(line_1_percent_randomsub)

            FivePercent.append(line_5_percent_randomsub)

            TenPercent.append(line_10_percent_randomsub)

            TwentyPercnt.append(line_20_percent_randomsub)


            GCBSOUT.append(line_GCBS)
            DOUBLESUNOUT.append(line_DOU)
            TLSMOUT.append(line_TLSM)
        '''
        write_1 = r'D:\Destop\file\科研相关\论文代码\PaperCode\Baselines\{}_Non_Specilized Method\1%_{}_{}.txt'.format(mode,mode, str(i))
        with open(write_1, 'w') as file2:
            for line in OnePercent:
                file2.write(line)
                file2.write('\n')

        write_5 = r'D:\Destop\file\科研相关\论文代码\PaperCode\Baselines\{}_Non_Specilized Method\5%_{}_{}.txt'.format(mode,mode, str(i))
        with open(write_5, 'w') as file2:
            for line in FivePercent:
                file2.write(line)
                file2.write('\n')

        write_10 = r'D:\Destop\file\科研相关\论文代码\PaperCode\Baselines\{}_Non_Specilized Method\10%_{}_{}.txt'.format(mode,mode, str(i))
        with open(write_10, 'w') as file2:
            for line in TenPercent:
                file2.write(line)
                file2.write('\n')

        write_20 = r'D:\Destop\file\科研相关\论文代码\PaperCode\Baselines\{}_Non_Specilized Method\20%_{}_{}.txt'.format( mode,mode, str(i))
        with open(write_20, 'w') as file2:
            for line in TwentyPercnt:
                file2.write(line)
                file2.write('\n')

        '''

        write_gcbs = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\baselines\GCBS_{}_{}.txt'.format(mode, mode, str(i))
        with open(write_gcbs, 'w') as file2:
            for line in GCBSOUT:
                file2.write(line)
                file2.write('\n')

        write_tlsm = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\baselines\TLSM_{}-{}.txt'.format(mode, mode,str(i))
        with open(write_tlsm, 'w') as file3:
            for line in TLSMOUT:
                file3.write(line)
                file3.write('\n')

        write = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\baselines\DOUBLESUB_{}-{}.txt'.format(mode, mode,str(i))
        with open(write, 'w') as file4:
            for line in DOUBLESUNOUT:
                file4.write(line)
                file4.write('\n')

        '''
        Files_Names = [write_1,write_5,write_10,write_20]
        for file_name in Files_Names:
            with open(path, 'r') as f1:
                linesori = f1.readlines()

            with open(file_name, 'r') as f2:
                linessc = f2.readlines()

            # CGd = cg_tm_kl.CG_b(linesori,linessc,len_sc=200)
            # Tmd = cg_tm_kl.Tmb(linesori, linessc, len_sc=200)

            # klds = cg_tm_kl.KLDoubleStrand(linessc,linesori)

            lineall = list(zip(linesori, linessc))

            All = []
            for l in lineall:
                lineori = list(''.join(l[0].split(' ')))
                linesc = list(''.join(l[1].split(' ')))
                num = 0
                for i in range(len(lineori)):
                    if lineori[i] != linesc[i]:
                        num += 1

                All.append(num / len(lineori))

            modificationrate = np.mean(np.array(All))

            print(modificationrate)
            # print('Tmd:',Tmd)
            # print('CGD:',CGd)
            # print('KLDS',klds)
        '''









