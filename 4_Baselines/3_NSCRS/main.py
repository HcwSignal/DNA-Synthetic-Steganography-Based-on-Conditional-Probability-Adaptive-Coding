import FindStartCondon
import random
def AdjustCandidatePoll_Hutong(GeneratedBase,CandidatePoll):
    AdjustedPoll  = []
    for Candidate in CandidatePoll:
        if len(GeneratedBase) < 2:
            AdjustedPoll.append(Candidate)
        else:
            CandidateTemp = GeneratedBase[len(GeneratedBase)-2 : ] + Candidate
            if FindStartCondon.FindStartCondon(CandidateTemp) == 0:
                AdjustedPoll.append(Candidate)

    return AdjustedPoll

def adjust_poll(x,vocabulary):
    #x为以生成的序列
    candidate_poll = ['A', 'T', 'C', 'G']
    if len(x) == 0:
        GeneratedBase = ''
    else:
        GeneratedBase = x[-1]

    Adjusted_poll = FindStartCondon.AdjustCandidatePoll(GeneratedBase=GeneratedBase,
                                                        CandidatePoll=candidate_poll)

    return prob_tensor

def hutong_baseline(single_line_num,total_num):
    #加载比特流
    with open(r'bit_stream.txt','r') as f1:
        bits = f1.readlines()

    #random.shuffle(bits[0])
    bits = list(bits[0])
    random.shuffle(bits)
    #候选碱基池
    candidate_poll = ['A','T','C','G']
    Generated_Base_Line = []
    i = 0
    while( len(Generated_Base_Line) < total_num):
        GeneratedBase = ''
        while( len(GeneratedBase) < single_line_num):
            # 碱基&二进制编码对应字典
            dict_base_bits = {}

            Adjusted_poll = AdjustCandidatePoll_Hutong(GeneratedBase=GeneratedBase, CandidatePoll=candidate_poll)
            if len(Adjusted_poll) == 4:
                dict_base_bits['00'] = 'A'
                dict_base_bits['01'] = 'T'
                dict_base_bits['10'] = 'C'
                dict_base_bits['11'] = 'G'
                GeneratedBase = GeneratedBase + dict_base_bits[str(bits[i]) + str(bits[i + 1])]
                i += 2
            elif len(Adjusted_poll) == 3:
                dict_base_bits['0'] = 'A'
                dict_base_bits['10'] = 'T'
                dict_base_bits['11'] = 'C'
                if str(bits[i]) == '0':
                    GeneratedBase = GeneratedBase + 'A'
                    i += 1
                else:
                    GeneratedBase = GeneratedBase + dict_base_bits[str(bits[i]) + str(bits[i + 1])]
                    i += 2
            elif len(Adjusted_poll) == 1:
                dict_base_bits[''] = 'C'
                GeneratedBase = GeneratedBase + 'C'

        Generated_Base_Line.append(GeneratedBase)

    return Generated_Base_Line

if __name__ == '__main__':
    ls = hutong_baseline(200,4200)
    with open(r'C:\Users\shmily\PycharmProjects\plot\hutong\847\hutong_200_3.txt','w') as f:
        for l in ls:
            f.write(l)
            f.write('\n')
