import numpy
import numpy as np
import pandas as pd
import scipy.stats
import random
import collections
import math
import os
import csv
import pandas
# 参数
import cg_tm_kl

len_ori = 48
len_sc = 48

beg_sc = 0
end_sc = 48

beg_ori = 0
end_ori = 5000
dictComplement = {}

dictComplement['A'] = 'T'
dictComplement['T'] = 'A'
dictComplement['C'] = 'G'
dictComplement['G'] = 'C'

def txt_process_sc(lines):
    ALL = []

    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        # temp = temp[:len_sc]
        # temp_out += temp
        # if len(temp) == len_sc:
        #    for i in range(0, len(temp) - 1, 2):
        #        temp1 += temp[i] + temp[i + 1] + ' '

        #    ALL.append(temp1)


        temp_out += temp
        for i in range(0, len(temp) - 1, 2):
            temp1 += temp[i] + temp[i + 1] + ' '

        ALL.append(temp1)

    # ALL = ALL[beg_sc:end_sc]
    return ALL, temp_out


def txt_process(lines):
    ALL = []
    num = []
    # numpy.random.shuffle(lines)
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        # len1 = len(temp)

        # num.append(len1)
        # most_num = np.argmax(np.bincount(num))
        # min_num = np.min(num)

        # for i in range(0, len(temp) - 1, 2):
        #   temp1 += temp[i] + temp[i + 1] + ' '

        # ALL.append(temp1)

        temp = temp[:len_ori]
        temp_out += temp
        if len(temp) == len_ori:
            for i in range(0, len(temp) - 1, 2):
                temp1 += temp[i] + temp[i + 1] + ' '

            ALL.append(temp1)

    ALL = ALL[beg_ori:end_ori]

    return ALL, temp_out


def str_to_list(lines):
    out = []
    for line in lines:
        line = line.split(" ")
        out += line

    return out



def pxy_doubleseq(line, sigle, two_base):
    BaseX = ['A', 'T', 'C', 'G']
    BaseY = ['A', 'T', 'C', 'G']
    BaseMartix = []
    DicSingleBaseNum = {}
    DicTwoBaseNum = {}
    DicP = {}

    for x in BaseX:
        for y in BaseY:
            bases = x + y
            BaseMartix.append(bases)

    for base, num in sigle:
        DicSingleBaseNum[base] = num

    for bases, num in two_base:
        DicTwoBaseNum[bases] = num

    for bases in BaseMartix:
        FirstBase = bases[0]
        SecondBase = bases[1]
        FirstBaseComplement = dictComplement[FirstBase]
        SecondBaseComplement = dictComplement[SecondBase]
        basesComplement = str(FirstBaseComplement + SecondBaseComplement)
        fXYDoubleseq = 0.5 * (DicTwoBaseNum[bases] + DicTwoBaseNum[basesComplement])
        fXDoubleseq  = 0.5 * (DicSingleBaseNum[FirstBase] + DicSingleBaseNum[FirstBaseComplement])
        fYDoubleseq  = 0.5 * (DicSingleBaseNum[SecondBase] + DicSingleBaseNum[SecondBaseComplement])

        PXY = (fXYDoubleseq * 2 * len(line)) / (fXDoubleseq * fYDoubleseq)

        DicP[bases] = PXY

    return DicP


def pxy(line, sigle, two_base):
    BaseX = ['A','T','C','G']
    BaseY = ['A', 'T', 'C', 'G']
    BaseMartix = []
    DicSingleBaseNum = {}
    DicTwoBaseNum = {}
    DicP = {}

    for x in BaseX:
        for y in BaseY:
            bases = x + y
            BaseMartix.append(bases)

    for base,num in sigle:
        DicSingleBaseNum[base] = num

    for bases , num in two_base:
        DicTwoBaseNum[bases] = num

    for bases in BaseMartix:
        FirstBase  = bases[0]
        SecondBase = bases[1]
        PXY = (DicTwoBaseNum[bases] * 2*len(line)) / (  DicSingleBaseNum[FirstBase] * DicSingleBaseNum[SecondBase])
        DicP[bases] = PXY

    return DicP



def KL(DICsc,DICori):
    sc = []
    ori = []
    for bases, Pxy in DICsc.items():
        sc.append(Pxy)

    for bases, Pxy in DICori.items():
        ori.append(Pxy)

    ori = ori / np.sum(ori)
    sc = sc / np.sum(sc)
    ZipScore = list(zip(sc,ori))

    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZipScore:
        TEMP = OriScore / ScScore
        KLD += -( ScScore * math.log(OriScore / ScScore,math.e ) )
        KLD1 += ScScore * np.log( ScScore / OriScore )

    OUT = scipy.stats.entropy(ori ,sc)
    '''
    x = [0.14285714, 0.04761905, 0.15873016, 0.07936508, 0.15873016, 0.06349206,0.11111111, 0.0952381,  0.12698413, 0.01587302]
    y = [0.0952381 , 0.07936508, 0.15873016, 0.01587302, 0.11111111, 0.14285714, 0.14285714, 0.0952381,  0.03174603, 0.12698413]
    ZIPxy = list(zip(x,y))
    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZIPxy:
        TEMP = OriScore / ScScore
        KLD += -(ScScore * math.log(OriScore / ScScore, math.e))
        KLD1 += ScScore * np.log(ScScore / OriScore)
    OUT = scipy.stats.entropy(x, y)
    '''
    return OUT

def SequenceComplement(lines):
    lines_list, line_str = txt_process_sc(lines)

    #line_sc = str_to_list(lines_list) # 原始生成数据，需要进行处理ori_two.txt

    #line_sc = line_sc[: len(line_sc) - 1]

    # str_temp_ori = list(''.join(line_ori))

    line_str_tolist = list(line_str)



    linecomplement = []

    for character in line_str_tolist:
        if len(character) == 1:
            complement = dictComplement[character]
            linecomplement.append(complement)
        else:
            complement0 = dictComplement[character[0]]
            complement1 = dictComplement[character[1]]
            temp = complement0 + complement1
            linecomplement.append(temp)

    linecomplement = ''.join(linecomplement)
    return linecomplement,line_str

def KLDoubleStrand(line_sc,line_ori):
    line_sc_complement,line_sc = SequenceComplement(line_sc)

    line_sc_doublesequence = line_sc + line_sc_complement

    temp = []
    temp.append(line_sc_doublesequence)

    line_sc_doublesplit, _ = txt_process_sc(temp)
    line_sc_doublesplit_list = line_sc_doublesplit[0].split(' ')[ : -1]

    str_temp_sc = list(_)

    singlebase = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)

    word_distribution_sc = sorted(collections.Counter(line_sc_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    sc_pxy = pxy_doubleseq(str_temp_sc, singlebase, word_distribution_sc)

    ############
    line_ori_complement, line_ori = SequenceComplement(line_ori)

    line_ori_doublesequence = line_ori + line_ori_complement

    temp = []
    temp.append(line_ori_doublesequence)

    line_ori_doublesplit, __ = txt_process_sc(temp)
    line_ori_doublesplit_list = line_ori_doublesplit[0].split(' ')[: -1]

    str_temp_ori = list(__)

    singlebase = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)

    word_distribution_ori = sorted(collections.Counter(line_ori_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    ori_pxy = pxy_doubleseq(str_temp_ori, singlebase, word_distribution_ori)

    kl_ = KL(sc_pxy, ori_pxy)

    return kl_

def KL_(line_sc, line_ori):
    line_sc_, all_sc = txt_process_sc(line_sc)
    line_ori_, all_ori = txt_process(line_ori)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)


    line_sc = line_sc[ : len(line_sc)-1]
    line_ori = line_ori[ : len(line_ori)-1]
    str_temp_sc = list(''.join(line_sc))
    singlebase = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)

    word_distribution_sc = sorted(collections.Counter(line_sc).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    sc_pxy = pxy(str_temp_sc, singlebase, word_distribution_sc)

    str_temp_ori = list(''.join(line_ori))

    singlebase = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)

    word_distribution_ori = sorted(collections.Counter(line_ori).items(), key=lambda x: x[1],
                                   reverse=True)  # 获得文件中各个单词的分布

    ori_pxy = pxy(str_temp_ori, singlebase, word_distribution_ori)

    kl_ = KL(sc_pxy,ori_pxy)

    return kl_


if __name__ == '__main__':
    path_ori = r'D:\Destop\seqs\888kb_ASM400647v1\read_4\OriginalData\read_4.txt'
    # with open(path_ori, "r") as f2:
    #    line_ori = f2.readlines()
    PathSc = []
    dirsSc = r'D:\Destop\seqs\888kb_ASM400647v1\ss_read_4\ss'
    for root, dirs, files in os.walk(dirsSc):
        for file in files:
            PathSc.append(os.path.join(root, file))

    raw_ori = cg_tm_kl.txt_process_sc_duo(path_ori,len_sc=200,beg_sc=0,end_sc=4500,PADDING=False,flex=0,num1=4)
    pd_kl = pd.DataFrame(columns={'name','kl','klds'})
    for p in PathSc:
        # with open(p, "r") as f1:
        #   line_sc = f1.readlines()

        raw_sc = cg_tm_kl.txt_process_sc_duo(p, len_sc=200, beg_sc=0, end_sc=4500, PADDING=False, flex=0,
                                              num1=4)
        kl = KL_(raw_sc, raw_ori)
        klds = KLDoubleStrand(raw_sc, raw_ori)
        pd_kl = pd_kl.append({'name':p[ p.rfind('\\') + 1 : -4], 'kl':kl,'klds':klds },ignore_index=True)
        print(pd_kl)


    pd_kl.to_csv(r'D:\Destop\seqs\888kb_ASM400647v1\ss_read_4\kl_2.csv')















