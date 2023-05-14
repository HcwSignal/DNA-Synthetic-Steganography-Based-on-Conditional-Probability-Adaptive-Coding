import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


def plt_KL(KL):
    width = 0.45

    KL_ORI = []
    KL_SC  = []
    KL_KEY = []
    for kl in KL:
        KL_ORI.append(kl[0])
        KL_SC.append(kl[1])
        KL_KEY.append(kl[2])

    index = np.arange(len(KL_ORI))
    plt.bar(index,KL_ORI,width,label = "KL_ORIGINAL",color = 'blue')
    plt.bar(index+ width,KL_SC,width,label = "KL_GENERATE",color= 'orange')
    plt.xticks(index+ width/2 ,labels=KL_KEY)
    plt.ylabel("rho")
    plt.legend(loc = 'best')
    plt.savefig(r'D:\Destop\seqs\new3_result\pic\1.jpg')


def txt_process_ori(dp_or,len_ori,beg_or,end_or,PADDING,flex):
    with open(dp_or,"r") as f1:
        lines = f1.readlines()

    ALL = []
    num = []
    #numpy.random.shuffle(lines)
    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        len1 = len(temp)

        num.append(len1)
        #most_num = np.argmax(np.bincount(num))
        #min_num = np.min(num)

        temp = temp[:len_ori]
        if PADDING == True:
            if len(temp) > (len_ori - flex):
                for i in range(0, len(temp) - 1, 2):
                    temp1 += temp[i] + temp[i + 1] + ' '

                ALL.append(temp1)
        else:
            if len(temp) == len_ori:
                for i in range(0, len(temp) - 1, 2):
                    temp1 += temp[i] + temp[i + 1] + ' '

                temp = temp1[:len(temp1)-1]
                ALL.append(temp)


    ALL = ALL[beg_or:end_or]

    return ALL
def find_bpn(path):
    with open(path,"r") as f1:
        lines = f1.readlines()

    bpn = 0
    for line in lines:
        temp = line.find("mean:rl") + len('mean:rl')
        if line.find("mean:rl") > 0 :
            bpn = line[temp:-1]

    return bpn

def txt_process_sc(dp_sc,len_sc,beg_sc,end_sc,PADDING,flex):
    with open(dp_sc,"r") as f1:
        lines = f1.readlines()

    ALL = []
    num = []
    #numpy.random.shuffle(lines)
    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        len1 = len(temp)
        num.append(len1)
        #most_num = np.argmax(np.bincount(num))
        #min_num = np.min(num)

        temp = temp[:len_sc]
        if PADDING == True:
            if len(temp) > (len_sc-flex):
                for i in range(0, len(temp) - 1, 2):
                    temp1 += temp[i] + temp[i + 1] + ' '

                ALL.append(temp1)

        else:
            if len(temp) == len_sc:
                for i in range(0, len(temp) - 1, 2):
                    temp1 += temp[i] + temp[i + 1] + ' '

                ALL.append(temp1)



    ALL = ALL[beg_sc:end_sc]

    return ALL

def shuffle(path):
    with open(path,'r') as f1:
        lines = f1.readlines()

    random.shuffle(lines)

    path_shuffle = r'C:\Users\Administrator\Desktop\seqs\zhuangzai_cnn\read_5-shuffle\read_5-shuffle.txt'
    with open(path_shuffle,'w') as f2:
        for line in lines:
            f2.write(line)
            f2.write('\n')

    return path_shuffle
def txt_process_sc_duo(dp_sc,len_sc,beg_sc,end_sc,PADDING,flex,num1,tiqu):
    with open(dp_sc,"r",encoding='gbk') as f1:
        lines = f1.readlines()
    #lines = lines[:3100]
    ALL = []
    num = []
    # 序列原始数据提取

    if tiqu == True:

        lines_ = []
        for line in lines:
            line = line.strip()
            if line.find('>') == 0:
                continue
            lines_.append(line)
        _ = ''.join(lines_)
        _ = _[:len(_) - len(_) % 200]
        lines = np.array(list(_)).reshape((-1, 200))


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
                for i in range(0, len(temp), num1):
                    temp1 += temp[i :i+ num1] + ' '

                ALL.append(temp1)

        else:
            if len(temp) == len_sc:
                for i in range(0, len(temp), num1):
                    temp1 += temp[i:i + num1] + ' '
                temp = temp1[:len(temp1)-1]
                ALL.append(temp)



    ALL = ALL[beg_sc:end_sc]

    return ALL

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

def CG_b(line_ori,line_sc):
    C_G_SUM_ori = 0
    C_G_PER_ori = []
    for line in line_ori:
        line_temp = line.strip().split(' ')
        line_temp = ''.join(line_temp)
        C_G_SUM_ori += C_G(line_temp)
        C_G_PER_ori.append(C_G(line_temp))

    C_G_SUM_sc = 0
    C_G_PER_sc = []
    # line_sc = np.array(list(line_sc[0])[:182952]).reshape(-1, 126)
    for line in line_sc:
        line_temp = line.strip().split(' ')
        line_temp = ''.join(line_temp)
        # line_temp = np.array(list(line_temp)[:182952]).reshape(-1,126)
        C_G_SUM_sc += C_G(line_temp)
        C_G_PER_sc.append(C_G(line_temp))
    C_G_PER_mean_ori = np.mean(C_G_PER_ori)  # 原序列的总体C-G含量
    C_G_PER_mean_SC  = np.mean(C_G_PER_sc)  # 生成序列的单句平均C-G含量

    CG_B = np.abs(C_G_PER_mean_SC-C_G_PER_mean_ori) / C_G_PER_mean_ori
    return CG_B,C_G_PER_mean_SC

def Tmb(line_ori,line_sc):
    tm_PER_ori = []
    for line in line_ori:
        line_temp = line.strip().split(' ')
        line_temp = ''.join(line_temp)
        try:
            tm_PER_ori.append(melting(line_temp))
            # tm_PER.append(melting(line))
        except:
            print(line_temp)
            # print(line)
            continue

    # print(tm_PER)

    tm_PER_sc = []
    # line_sc = np.array(list(line_sc[0])[:182952]).reshape(-1, 126)
    for line in line_sc:
        line_temp = line.strip().split(' ')

        line_temp = ''.join(line_temp)
        # line_temp = np.array(list(line_temp)[:182952]).reshape(-1,126)

        try:
            tm_PER_sc.append(melting(line_temp))
        except:
            print(line_temp)
            continue

    tm_mean_ori = np.mean(tm_PER_ori)
    tm_mean_sc = np.mean(tm_PER_sc)

    Tmb = np.abs(tm_mean_ori-tm_mean_sc) / tm_mean_ori

    return Tmb,tm_mean_sc

if __name__ == '__main__':
    len_ori = 198
    len_sc = 198
    beg_sc  = 0
    beg_ori = 0
    end_sc  = 5000
    end_ori = 5000


    dp_sc = r"C:\Users\Administrator\Desktop\rnn-stega_pytorch_\log\line_888kb_5\rnn_fxy1-9_huf_fxy_0.txt"
    dp_or = r'C:\Users\Administrator\Desktop\line_660kb\read_1.txt'

    line_sc  = txt_process_sc(dp_sc,len_sc,beg_sc,end_sc) #原始生成数据，需要进行处理ori_two.txt
    line_ori = txt_process_ori(dp_or,len_ori,beg_ori,end_ori)


    C_G_SUM = 0
    C_G_PER = []
    tm_PER = []
    for line in line_ori:
        line_temp = line.strip().split(' ')
        line_temp = ''.join(line_temp)
        C_G_SUM += C_G(line_temp)
        C_G_PER.append(C_G(line_temp))
        # C_G_SUM += C_G(line)
        # C_G_PER.append(C_G(line))
        try:
            tm_PER.append(melting(line_temp))
            # tm_PER.append(melting(line))
        except:
            print(line_temp)
            # print(line)
            continue

    #print(tm_PER)

    C_G_SUM_sc = 0
    C_G_PER_sc = []
    tm_PER_sc = []
    # line_sc = np.array(list(line_sc[0])[:182952]).reshape(-1, 126)
    for line in line_sc:
        line_temp = line.strip().split(' ')

        line_temp = ''.join(line_temp)
        #line_temp = np.array(list(line_temp)[:182952]).reshape(-1,126)
        C_G_SUM_sc += C_G(line_temp)
        C_G_PER_sc.append(C_G(line_temp))
        try:
            tm_PER_sc.append(melting(line_temp))
        except:
            continue

    tm_mean_ori = np.mean(tm_PER)
    tm_mean_sc  = np.mean(tm_PER_sc)

    C_G_PER_mean = np.mean(C_G_PER) #原序列的总体C-G含量
    C_G_PER_mean_SC = np.mean(C_G_PER_sc) #生成序列的单句平均C-G含量

    print('tm_mean_ori:',tm_mean_ori)
    print('tm_mean_sc:',tm_mean_sc)
    print("C_G_MEAN_SC:",C_G_PER_mean_SC)
    print("C_G_MEAN_ORI:",C_G_PER_mean)
    print('tm_bias:',(np.abs(tm_mean_ori-tm_mean_sc) / tm_mean_ori)*100,"%")
    print('CG_BIAS:',(np.abs(C_G_PER_mean_SC-C_G_PER_mean) / C_G_PER_mean)*100,"%")
    print(CG_b(line_ori,line_sc))
    print(Tmb(line_ori,line_sc))
    #print(C_G_PER)
    #print(C_G_SUM)
    #print(C_G_SUM_sc)