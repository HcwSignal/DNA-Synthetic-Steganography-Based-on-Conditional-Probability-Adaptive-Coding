import numpy
import numpy as np
import scipy.stats
import random
import collections
import math
# 参数
len_ori = 48
len_sc = 48

beg_sc = 0
end_sc = 48

beg_ori = 0
end_ori = 5000


def txt_process_sc(lines):
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

        #temp = temp[:len_sc]
        temp_out += temp
        if len(temp) > 10:
            for i in range(0, len(temp) - 1, 2):
                temp1 += temp[i] + temp[i + 1] + ' '

            ALL.append(temp1)

    ALL = ALL[beg_sc:end_sc]

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

        #temp = temp[:len_ori]
        temp_out += temp
        if len(temp) > 10:
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


# pxy函数一段序列中十六个二核苷酸的pxy值
# input为一段连续str，output为十六个二核苷酸的pxy值
def pxy1(line):
    len_all = len(line)
    temp_ = collections.Counter(list(line)).items()
    dict_base = {}
    for key, value in temp_:
        dict_base[key] = value  # 求A\T\C\G分别的含量

    temp = ''
    for i in range(0, len(line) - 1, 2):
        temp += line[i] + line[i + 1] + ' '
    list_temp = temp.split(' ')
    temp_2 = collections.Counter(list_temp).items()  # 建立二核苷酸的统计数据——第一次读取

    line_2 = line[1:len(line) - 1]  # 错位读第二次
    temp = ''
    for i in range(0, len(line_2) - 1, 2):
        temp += line_2[i] + line_2[i + 1] + ' '
    list_temp_ = temp.split(' ')
    temp_2_1 = collections.Counter(list_temp_).items()  # 建立二核苷酸的统计数据——第二次读取

    dict_2 = {}
    dict_1 = {}

    for key, value in temp_2:
        dict_2[key] = value

    for key, value in temp_2_1:
        dict_1[key] = value

    dict_all = {}
    for key, value in temp_2:
        dict_all[key] = dict_2[key] + dict_1[key]  # 将统计的数据合并

    out = {}
    for key, value in temp_2:
        try:
            first = key[0]
            second = key[1]
            out[key] = (value * len_all) / (dict_base[first] * dict_base[second])  # 计算pxy
        except:
            return out


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
        PXY = (DicTwoBaseNum[bases] * len(line)) / (  DicSingleBaseNum[FirstBase] * DicSingleBaseNum[SecondBase])
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


def KL_(line_sc, line_ori):
    line_sc_, all_sc = txt_process_sc(line_sc)
    line_ori_, all_ori = txt_process(line_ori)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)

    xy_sc = collections.Counter(line_sc)
    xy_ori = collections.Counter(line_ori)

    KL_num = []
    for key_ori, value_ori in xy_ori.items():
        KL_TEMP = []
        for key_sc, value_sc in xy_sc.items():
            if key_sc == key_ori:
                KL_TEMP.append(value_ori)
                KL_TEMP.append(value_sc)
                KL_num.append(KL_TEMP)
                continue

    KL_num = KL_num[:len(KL_num) - 1]

    # print(kl)
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
    path_ori = r'D:\Destop\seqs\KLtest\1.txt'
    path_sc = r'D:\Destop\seqs\KLtest\2.txt'
    with open(path_sc, "r") as f1:
        line_sc = f1.readlines()

    with open(path_ori, "r") as f2:
        line_ori = f2.readlines()

    kl = KL_(line_sc, line_ori)
    print(kl)





