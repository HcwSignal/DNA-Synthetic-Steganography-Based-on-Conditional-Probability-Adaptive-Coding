import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import scipy.stats
import math
dictComplement = {}
dictComplement['A'] = 'T'
dictComplement['T'] = 'A'
dictComplement['C'] = 'G'
dictComplement['G'] = 'C'
def txt_process_sc_FromKl(lines,len_sc,divied):
    ALL = []
    num1 = divied
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        temp = temp[:len_sc]
        temp_out += temp
        if len(temp) == len_sc:
            for i in range(0, len(temp), num1):
                temp1 += temp[i:i + num1] + ' '

            ALL.append(temp1)

    return ALL, temp_out
def BaseExtraaction(dp_sc,len_single,batchsize):
    with open(dp_sc,"r") as f1:
        lines = f1.readlines()

    ALL = []

    for line in lines:
        if len(line.split('>')) > 1:
            continue
        temp = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        # len1 = len(temp)
        # num.append(len1)
        ALL.append(temp)

    AllSeq = ''.join(ALL)

    # np.random.shuffle(list(AllSeq))

    Numhang = len(AllSeq) -  len(AllSeq) % len_single

    listAllSeq = (np.array(list(AllSeq))[ 0:Numhang]).reshape((-1,len_single))

    ALLout = []
    for line in listAllSeq:
        line = ''.join(line)
        ALLout.append(line)

    End = []
    for i in range(0,len(ALLout)+1):
        if (i * 0.9) % batchsize == 0 and (i * 0.1) % batchsize == 0:
            End.append(i)

    # EndSimple = len(ALLout) - len(ALLout) % batchsize
    ALLout = ALLout [ : End[-1]]

    return ALLout
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
def txt_process(lines,length,beg,end):
    ALL = []
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        temp = temp[:length]
        temp_out += temp
        if len(temp) == length:
            for i in range(0, len(temp) - 1, 2):
                temp1 += temp[i] + temp[i + 1] + ' '

            ALL.append(temp1)

    ALL = ALL[beg:end]

    return ALL, temp_out

def txt_process_sc_duo(dp_sc,len_sc,beg_sc,end_sc,PADDING,flex,devide_num):
    with open(dp_sc,"r",encoding='utf-8',errors='ignore') as f1:
        lines = f1.readlines()

    ALL = []
    num = []

    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        len1 = len(temp)
        num.append(len1)

        temp = temp[:len_sc]
        if PADDING == True:
            if len(temp) > (len_sc-flex):
                for i in range(0, len(temp), devide_num):
                    temp1 += temp[i :i+ devide_num] + ' '

                ALL.append(temp1)

        else:
            if len(temp) == len_sc:
                for i in range(0, len(temp), devide_num):
                    temp1 += temp[i:i + devide_num] + ' '
                temp = temp1[:len(temp1)-1]
                ALL.append(temp)



    ALL = ALL[beg_sc:end_sc]

    return ALL

def find_bpn(path):
    with open(path,"r") as f1:
        lines = f1.readlines()

    bpn = 0
    flag = 0
    for line in lines:
        temp = line.find("mean:rl") + len('mean:rl')
        if line.find("mean:rl") > 0 :
            bpn = line[temp:-1]
            flag = 1

    if flag == 0:
        BPN = []
        for line in lines:
            temp = line.find("rl:") + len('rl:')
            if line.find("rl:") > 0:
                BPN.append(float(line[temp : -1 ]))

        bpn = np.mean(np.array(BPN))

    return bpn

def str_to_list(lines):
    out = []
    for line in lines:
        line = line.split(" ")
        out += line

    return out

def C_G(line):
    num = 0
    for i in range(len(line)):
        if (line[i] == 'C') or (line[i] == 'G'):
            num += 1

    return num / len(line)

def melting(line):
    dic_temp = {}
    for i in range(len(line)):
        dic_temp[line[i]] = dic_temp.get(line[i],0) + 1

    nG = dic_temp.get('G')
    nA = dic_temp.get('A')
    nC = dic_temp.get('C')
    nT = dic_temp.get('T')

    tm = 64.9 + 41*( (nG + nC -16.4) / (nG + nA + nT + nC) )


    return tm

def CG_b(line_ori,line_sc,len_sc):
    line_sc_, all_sc = txt_process_sc_FromKl(line_sc,len_sc,divied=1)
    line_ori_, all_ori = txt_process_sc_FromKl(line_ori,len_sc,divied=1)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)

    CGORI = C_G(line_ori)
    CGSC  = C_G(line_sc)

    CGBias = np.abs( CGORI - CGSC) / CGORI

    return CGBias * 100

def Tmb(line_ori,line_sc,len_sc):
    line_sc_, all_sc = txt_process_sc_FromKl(line_sc, len_sc, divied=1)
    line_ori_, all_ori = txt_process_sc_FromKl(line_ori, len_sc, divied=1)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)

    TMORI = melting(line_ori)
    TMSC = melting(line_sc)

    TMBias = np.abs(TMORI - TMSC) / TMORI

    return TMBias * 100
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
    lines_list, line_str = txt_process(lines,length=200,beg=0,end=3300)

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

def split_data(path,dividenum):
    with open(path,"r") as f1:
        lines = f1.readlines()

    temp = ''
    all = ''
    for line in lines:
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

    stop = len(temp) - len(temp) % dividenum

    temp = temp[ : stop]

    for i in range(0,len(temp),dividenum):
        all += temp[i : i + dividenum] + ' '

    temp = temp[ : 22400]

    l = list(temp)
    out = np.array(list(temp)).reshape((175,-1))

    return all,out

