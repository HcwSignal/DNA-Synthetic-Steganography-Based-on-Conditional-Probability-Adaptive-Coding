import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def count_1(lines):
    num_A , num_T , num_C , num_G  = 0, 0, 0, 0
    num_AA, num_AT, num_AC, num_AG = 0, 0, 0, 0
    num_TA, num_TT, num_TC, num_TG = 0, 0, 0, 0
    num_CA, num_CT, num_CC, num_CG = 0, 0, 0, 0
    num_GA, num_GT, num_GC, num_GG = 0, 0, 0, 0

    for i in range(len(lines)):
        if lines[i] == 'A':
            num_A = num_A + 1
            continue
        elif lines[i] == 'T':
            num_T = num_T + 1
            continue
        if lines[i] == 'C':
            num_C = num_C + 1
            continue
        elif lines[i] == 'G':
            num_G = num_G + 1


    for i in range(len(lines) - 1):
        if ((lines[i] == 'A') and (lines[i + 1] == 'A')):
            num_AA = num_AA + 1
            continue
        elif ((lines[i] == 'A') and (lines[i + 1] == 'T')):
            num_AT = num_AT + 1
            continue
        if ((lines[i] == 'A') and (lines[i + 1] == 'C')):
            num_AC = num_AC + 1
            continue

        elif ((lines[i] == 'A') and (lines[i + 1] == 'G')):
            num_AG = num_AG + 1




    for i in range(len(lines) - 1):
        if ((lines[i] == 'T') and (lines[i + 1] == 'A')):
            num_TA = num_TA + 1
            continue

        elif ((lines[i] == 'T') and (lines[i + 1] == 'T')):
            num_TT = num_TT + 1
            continue

        if ((lines[i] == 'T') and (lines[i + 1] == 'C')):
            num_TC = num_TC + 1
            continue

        elif ((lines[i] == 'T') and (lines[i + 1] == 'G')):
            num_TG = num_TG + 1




    for i in range(len(lines) - 1):
        if ((lines[i] == "G") and (lines[i + 1] == 'A')):
            num_GA = num_GA + 1
            continue

        elif ((lines[i] == 'G') and (lines[i + 1] == 'T')):
            num_GT = num_GT + 1
            continue

        if ((lines[i] == 'G') and (lines[i + 1] == 'C')):
            num_GC = num_GC + 1
            continue

        elif ((lines[i] == 'G') and (lines[i + 1] == 'G')):
            num_GG = num_GG + 1




    for i in range(len(lines) - 1):
        if ((lines[i] == "C") and (lines[i + 1] == "A")):
            num_CA = num_CA + 1
            continue

        elif ((lines[i] == 'C') and (lines[i + 1] == 'T')):
            num_CT = num_CT + 1
            continue

        if ((lines[i] == 'C') and (lines[i + 1] == 'C')):
            num_CC = num_CC + 1
            continue

        elif ((lines[i] == 'C') and (lines[i + 1] == 'G')):
            num_CG = num_CG + 1



    Start = []

    num_ATG,num_TTA,num_TAG,num_TGA= 0,0,0,0
    for i in range(0,len(lines) - 2):
        if ((lines[i] == "A") and (lines[i + 1] == "T") and (lines[i + 2] == "G")):
            num_ATG = num_ATG + 1
            continue

        elif ((lines[i] == "T") and (lines[i + 1] == "T") and (lines[i + 2] == "A")):
            num_TTA = num_TTA + 1
            continue

        elif ((lines[i] == "T") and (lines[i + 1] == "A") and (lines[i + 2] == "G")):
            num_TAG = num_TAG + 1
            continue

        elif ((lines[i] == "T") and (lines[i + 1] == "G") and (lines[i + 2] == "A")):
            num_TGA = num_TGA + 1


    Start.append(num_ATG)
    P_AA, P_AT, P_AC, P_AG, P_TA, P_TC, P_TG, P_TT, P_CA, P_CC, P_CG, P_CT, P_GA, P_GC, P_GG, P_GT = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

    if num_A != 0:
        P_AA = num_AA / (num_A * num_A) * len(lines)
        if num_T != 0 :
            P_AT = num_AT / (num_A * num_T) * len(lines)
        if num_C != 0:
                P_AC = num_AC / (num_A * num_C) * len(lines)
        if num_G != 0:
                    P_AG = num_AG / (num_A * num_G) * len(lines)

    if num_T != 0:
        P_TT = num_TT / (num_T * num_T) * len(lines)
        if num_A != 0:
            P_TA = num_TA / (num_T * num_A) * len(lines)
        if num_C != 0:
                P_TC = num_TC / (num_T * num_C) * len(lines)
        if num_G != 0:
                    P_TG = num_TG / (num_T * num_G) * len(lines)

    if num_G != 0:
        P_GG = num_GG / (num_G * num_G) * len(lines)
        if num_A != 0:
            P_GA = num_GA / (num_G * num_A) * len(lines)
        if num_C != 0:
                P_GC = num_GC / (num_G * num_C) * len(lines)
        if num_T != 0:
                    P_GT = num_GT / (num_G * num_T) * len(lines)

    if num_C != 0:
        P_CC = num_CC / (num_C * num_C) * len(lines)
        if num_A != 0:
            P_CA = num_CA / (num_C * num_A) * len(lines)
        if num_T != 0:
                P_CT = num_CT / (num_C * num_T) * len(lines)
        if num_G != 0:
                    P_CG = num_CG / (num_C * num_G) * len(lines)

    return P_AA, P_TT, P_AT, P_CA, P_TG, P_GA, P_TC, P_TA, P_AG, P_CT, P_AC, P_GT, P_GC, P_CC, P_GG, P_CG

Balance = 0
def KL(p,q):
    KL_1 = scipy.stats.entropy(p, q)
    return KL_1

def convert(line,voc):
    str_out = ''
    for i in range(len(line)):
        str_out += voc.i2w[line[i]]
    return str_out

def m_rep(prob,kl,rep):
    prob = prob.cpu().numpy()
    prob_rep = []
    zero = [0,0,0,0]
    for j in range(rep):
        prob_cl = prob[j][4:]
        prob_fal = []

        for i in range(16):
            out = prob_cl[i] - Balance * kl[j][i]
            if out < 0:
                out = 0
            prob_fal.append(out)
        prob_fal = zero + prob_fal

        prob_rep.append(prob_fal)

    prob_rep = np.array(prob_rep)
    Nan1 = np.where(np.isnan(prob_rep),0,prob_rep)

    prob_out = torch.from_numpy(Nan1).to(device)

    return prob_out

def m(prob,kl):
    prob = prob.cpu().numpy()
    prob_cl = prob[0][4:]
    prob_fal = []
    for i in range(16):
        out = prob_cl[i] - Balance * kl[i]
        if out < 0:
            out = 0
        prob_fal.append(out)

    prob_out = np.pad(prob_fal, (4, 0), 'constant', constant_values=(0, 0))
    prob_out = np.array(prob_out).reshape((1, 20))
    prob_out = torch.from_numpy(prob_out).to(device)

    return prob_out


def KL_gradient_rep(line, tol, voc, rep):
    KL_rep = []
    line1 = line.cpu().numpy()
    Class = ['TT', 'AA', 'AT', 'CA', 'TG', 'GA', 'TA', 'TC', 'AG', 'AC', 'CT', 'GT', 'GC', 'GG', 'CC', 'CG']
    for j in range(rep):
        KL_add_ALL = []
        line_now = line1[j][1:]
        for i in range(4, 20):
            next_w = voc.i2w[i]
            str_now = convert(line_now, voc)
            str_add = str_now + next_w
            P_TT, P_AA, P_AT, P_CA, P_TG, P_GA, P_TA, P_TC, P_AG, P_AC, P_CT, P_GT, P_GC, P_GG, P_CC, P_CG = count_1(str_now)
            now_ = [P_TT, P_AA, P_AT, P_CA, P_TG, P_GA, P_TA, P_TC, P_AG, P_AC, P_CT, P_GT, P_GC, P_GG, P_CC, P_CG]
            dic_now = dict(zip(Class, now_))
            now = []
            for j in range(4, 20):
                for key in dic_now:
                    if key == voc.i2w[j]:
                        now.append(dic_now[key])
                        continue

            P_TT, P_AA, P_AT, P_CA, P_TG, P_GA, P_TA, P_TC, P_AG, P_AC, P_CT, P_GT, P_GC, P_GG, P_CC, P_CG= count_1(str_add)
            add_ = [P_TT, P_AA, P_AT, P_CA, P_TG, P_GA, P_TA, P_TC, P_AG, P_AC, P_CT, P_GT, P_GC, P_GG, P_CC, P_CG]
            dic_add = dict(zip(Class, add_))
            add = []
            for j in range(4, 20):
                for key in dic_add:
                    if key == voc.i2w[j]:
                        add.append(dic_add[key])
                        continue


            '''
            AA, TT, AT, CA, TG, GA, TC, TA, AG, CT, AC, GT, GC, CC, GG, CG = count_1(str_add)
            ALL_add = np.vstack((np.array(AA), np.array(TT), np.array(AT), np.array(CA),
                                 np.array(TG), np.array(GA), np.array(TC), np.array(TA),
                                 np.array(AG), np.array(CT), np.array(AC), np.array(GT),
                                 np.array(GC), np.array(CC), np.array(GG), np.array(CG)))

            NUM_each = ALL_add.ravel()
            add = NUM_each.astype(np.float)
            '''
            KL_single = KL(add, tol) - KL(now, tol)

            KL_add_ALL.append(KL_single)

        KL_rep.append(KL_add_ALL)

    Nan1 = np.where(np.isnan(KL_rep), 0, KL_rep)
        
    return Nan1


def KL_gradient(line, tol, voc):
    line1 = line.cpu().numpy()
    KL_add_ALL = []
    line_now = line1[0][1:]
    for i in range(4, 20):
        next_w = voc.i2w[i]
        str_now = convert(line_now, voc)
        str_add = str_now + next_w
        AA, TT, AT, CA, TG, GA, TC, TA, AG, CT, AC, GT, GC, CC, GG, CG = count_1(str_now)
        ALL_now = np.vstack((np.array(AA), np.array(TT), np.array(AT), np.array(CA),
                             np.array(TG), np.array(GA), np.array(TC), np.array(TA),
                             np.array(AG), np.array(CT), np.array(AC), np.array(GT),
                             np.array(GC), np.array(CC), np.array(GG), np.array(CG)))

        NUM_each = ALL_now.ravel()
        now = NUM_each.astype(np.float)

        AA, TT, AT, CA, TG, GA, TC, TA, AG, CT, AC, GT, GC, CC, GG, CG = count_1(str_add)
        ALL_add = np.vstack((np.array(AA), np.array(TT), np.array(AT), np.array(CA),
                             np.array(TG), np.array(GA), np.array(TC), np.array(TA),
                             np.array(AG), np.array(CT), np.array(AC), np.array(GT),
                             np.array(GC), np.array(CC), np.array(GG), np.array(CG)))

        NUM_each = ALL_add.ravel()
        add = NUM_each.astype(np.float)
        KL_single = KL(add, tol) - KL(now, tol)

        KL_add_ALL.append(KL_single)


    return KL_add_ALL


def KL_gradient_rep_recover(line, tol, voc, rep):
    KL_rep = []
    line1 = line.cpu().numpy()
    Class = ['TT', 'AA', 'AT', 'CA', 'TG', 'GA', 'TA', 'TC', 'AG', 'AC', 'CT', 'GT', 'GC', 'GG', 'CC', 'CG']
    for j in range(rep):
        line_now = line1[j][1:]
        for u in range(1,len(line)):
            line_jianqu = line[:len(line) - u]
            add_1 = []
            KL_ALL = []
            for i in range(4,20):
                next_word = voc.i2w[i]
                str_add = line_jianqu + next_word
                dict_2 = {}
                dict_1 = {}
                for key in str_add:
                    dict_2[key] = dict_2.get(key, 0) + 1

                temp = 0
                for key, value in dict_2.items():
                    temp += value

                for key1 in Class:
                    dict_1[key1] = dict_2[key1] / temp
                    add_1.append(dict_2[key1 / temp])


                KL_ = KL(add_1,tol)
                KL_ALL.append(KL_)












class LM(nn.Module):
	def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate):
		super(LM, self).__init__()
		self._cell = cell

		self.embedding = nn.Embedding(vocab_size, embed_size)#vocab_size代表词表中有几个词，embed_size代表词维度
		if cell == 'rnn':
			self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'gru':
			self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'lstm':
			self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		else:
			raise Exception('no such rnn cell')

		self.output_layer = nn.Linear(hidden_dim, vocab_size)
		self.log_softmax = nn.LogSoftmax(dim=2)

	def forward(self, x, logits=False):
		x = x.long()
		_ = self.embedding(x)  # 对于输入的词向量，将其映射到高维空间
		_ = _.permute(1, 0, 2)  # 将tensor换维度，将第一个与第二个维度参数互换
		h_all, __ = self.rnn(_)
		h_all = h_all.permute(1, 0, 2)
		_ = self.output_layer(h_all)  # 实现线性变换
		if logits:
			return _
		else:
            
			return self.log_softmax(_)  #平滑处理

	def sample_beg(self, x):
		log_prob = self.forward(x)
		prob = torch.exp(log_prob)[:, -1, :]
		# p, i = prob.sort(descending=True)
		# self.p = p
		prob[:, 0:3] = 0# 这里取零的意思应该是将起始标志得到的数值置于0，以免在随意取样
		# 函数中取到起始标志0，但是这里应该为prob[:, 0] = 0才对?

		prob = prob / prob.sum()

		# xout = (torch.multinomial(prob, 1)).item()
		# prob_out = prob.cpu().numpy()
		# p = prob_out[0][xout]
		return torch.multinomial(prob, 1)

	def sample_rep(self, x, Tol,vocabulary,rep):
		K_gra_16 = []
		log_prob = self.forward(x)
		prob = torch.exp(log_prob)[:, -1, :]
		# p, i = prob.sort(descending=True)
		# self.p = p
		prob[:, 1] = 0# 这里取零的意思应该是将起始标志得到的数值置于0，以免在随意取样
		# 函数中取到起始标志0，但是这里应该为prob[:, 0] = 0才对?
		prob = prob / prob.sum()
		sorted11,index = torch.sort(prob)
		K_gra_16 = KL_gradient_rep(x, Tol, vocabulary,rep)
		sum_k = sum(sum(np.abs(K_gra_16)))
		K_gra_16_ = K_gra_16 / sum_k
		pp = m_rep(prob, K_gra_16_,rep)
		# print('pp:',pp)
		# print('nan values:', torch.sum(torch.isnan(pp)).item())
		
		# os.system("pause")


		return torch.multinomial(pp, 1)

	def sample_1(self, x, Tol,vocabulary):
		K_gra_16 = []
		log_prob = self.forward(x)
		prob = torch.exp(log_prob)[:, -1, :]
		# p, i = prob.sort(descending=True)
		# self.p = p
		prob[:, 1] = 0# 这里取零的意思应该是将起始标志得到的数值置于0，以免在随意取样
		# 函数中取到起始标志0，但是这里应该为prob[:, 0] = 0才对?
		prob = prob / prob.sum()
		K_gra_16 = KL_gradient(x, Tol, vocabulary)
		K_gra_16 = K_gra_16 / sum(np.abs(K_gra_16))

		pp = m(prob, K_gra_16)

		return torch.multinomial(pp, 1)

