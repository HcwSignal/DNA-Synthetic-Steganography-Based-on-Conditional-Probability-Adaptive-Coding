# -*- coding: utf-8 -*-
import numpy as np
import random
import cg_tm_kl
import os
from collections import Counter
def ComplemantarySequence(Bases):
    BasesComplementary = {}
    BasesComplementary['A'] = 'T'
    BasesComplementary['T'] = 'A'

    BasesComplementary['G'] = 'C'
    BasesComplementary['C'] = 'G'
    FinalSeq = ''
    for base in Bases:
        FinalSeq = BasesComplementary[base] + FinalSeq

    return FinalSeq

def FindStartCondon(line):
    Finding = 0
    StartCondon = ['ATG','CTG','TTG']
    BaseSequence = list(line)
    ComSeq = ComplemantarySequence(BaseSequence)
    for i in range(len(BaseSequence)-2):
        Condon = BaseSequence[i] + BaseSequence[i+1] + BaseSequence[i+2]
        Condon_C = ComSeq[i] + ComSeq[i + 1] + ComSeq[i + 2]
        if (Condon in StartCondon) or (Condon_C in StartCondon):
            Finding = 1
            #print('StartCondon Index:',i)
            break


    return Finding

def AdjustCandidatePoll(GeneratedBase,CandidatePoll):
    #GeneratedBase = 'ATCGA'
    AdjustedPoll  = {}
    for Candidate in list(CandidatePoll):
        CandidateTemp = GeneratedBase[len(GeneratedBase)-2 : ] + Candidate
        if FindStartCondon(CandidateTemp) == 0:
            AdjustedPoll[Candidate] = CandidatePoll[Candidate]

    return AdjustedPoll

def main(read_path,len_sc):
    #read_path = r'D:\Destop\code\660_PSEUDO_SEQ_TEMP\adg_fxy3-82-34_0.txt'
    #with open(read_path,'rb') as f1:
    #    lines_original = f1.readlines()

    lines = cg_tm_kl.txt_process_sc_duo(read_path,len_sc=len_sc,beg_sc=0,end_sc=1000000,PADDING=False,flex=0,devide_num=len_sc)
    ExitStartCondonNum = 0
    NoStCon = []
    AllCon = []
    for line in lines:
        ExitStartCondonNum = ExitStartCondonNum + FindStartCondon(line)
        AllCon.append(line)
        if FindStartCondon(line) == 0:
            NoStCon.append(line)

    ExitStartCondonRatio = ExitStartCondonNum / len(lines)
    #print(ExitStartCondonRatio)
    #print(ExitStartCondonNum)

    conunt_NoCon = Counter(NoStCon)
    conunt_AllCon = Counter(AllCon)
    return ExitStartCondonRatio,len(lines),len(conunt_NoCon),len(conunt_AllCon)

def GenerateBaseWord(number,demension):
    Base = ['A','T','C','G']
    GenerateBaseWords = []
    for i in range(0,number):
        GeneBase = ''
        for j in range(0,demension):
            GeneBase = GeneBase + Base[random.randint(0,3)]

        GenerateBaseWords.append(GeneBase)

    return GenerateBaseWords

if __name__ == '__main__':
    '''
    CandidatePollLength = 10
    CandidateDemension = 3
    Prob = []
    CandidatePoll = {}
    for i in range(0,CandidatePollLength):
        Prob.append(random.random())

    ProbNormalition = [P / sum(Prob) for P in Prob]

    GenerateBaseWords = GenerateBaseWord(CandidatePollLength,CandidateDemension)

    for i in range(0,CandidatePollLength):
        CandidatePoll[GenerateBaseWords[i]] = Prob[i]

    AdjustCandidatePoll(GeneratedBase = 'ATCGAT',CandidatePoll= CandidatePoll)
    '''
    ExitStartCondonRatio,l,le,len = main(r'D:\Destop\1\2\adg_fxy3-54-39_0.txt',len_sc=198)
    print(ExitStartCondonRatio,'',l,le,len)







