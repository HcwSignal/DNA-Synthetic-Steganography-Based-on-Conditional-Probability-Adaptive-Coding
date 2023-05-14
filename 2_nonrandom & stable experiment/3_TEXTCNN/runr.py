import torch
from torch import nn
import torch.optim as optim
# import jieba
import argparse
import sys
from logger import Logger
# from torchsummary import summary
#import matplotlib.pyplot as plt
import numpy as np

import data
import textrnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 8 : 0,1,...7  集群，卡很多

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x : x.lower() == 'true')
    parser.add_argument("--filename", type=str, default='bal+=0.2')  #日志文件路径
    #parser.add_argument("--filename_pos", type=str, default='') #正例子路径
    #parser.add_argument("--filename_neg", type=str, default='ori_20%_two') #反例子路径
    parser.add_argument("--epoch", type=int, default=400)  # 设置了迭代200次
    parser.add_argument("--stop", type=int, default=300)
    args = parser.parse_args(sys.argv[1:])
    return args
args = get_args()
# logger
log_file = "./log/5k_per100/rnn_{}.txt".format(args.filename)
logger = Logger(log_file)

Train_loss = []
Test_loss = []
Train_acc = []
Test_acc = []

def main(data_helper):
	# ======================
	# 超参数
	# ======================
	CELL = "lstm"            # rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm
	#CELL = "bi-rnn"
	BATCH_SIZE = 50
	# BATCH_SIZE = 64
	EMBED_SIZE = 128
	HIDDEN_DIM = 128
	#HIDDEN_DIM = 196
	NUM_LAYERS = 1
	CLASS_NUM = 2
	# DROPOUT_RATE = 0.2
	DROPOUT_RATE = 0.2
	EPOCH = args.epoch
	#LEARNING_RATE = 0.01
	LEARNING_RATE = 0.01
	SAVE_EVERY = 20
	STOP = args.stop

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			logger.info("{0:15}   ".format(var))
			logger.info(all_var[var])
	print()

	# ======================
	# 数据
	# ======================

	# ======================
	# 构建模型
	# ======================
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda")
	model = textrnn.TextRNN(
		cell=CELL,
		vocab_size=data_helper.vocab_size,
		embed_size=EMBED_SIZE,
		hidden_dim=HIDDEN_DIM,  # 宏定义
		num_layers=NUM_LAYERS,
		class_num=CLASS_NUM,  # 2
		dropout_rate=DROPOUT_RATE  # 0.2
	)
	model.to(device)
# 	summary(model, (20,))
	criteration = nn.CrossEntropyLoss()  # 损失函数，交叉熵
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  #优化器
	#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	best_acc = 0
	early_stop = 0

	# ======================
	# 训练与测试
	# ======================
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)  # BATCH_SIZE = 50
		generator_test = data_helper.test_generator(BATCH_SIZE)
		train_loss = []
		train_acc = []
		while True:
			try:
				text, label = generator_train.__next__()
			except:
				break
			optimizer.zero_grad()
			y = model(torch.from_numpy(text).long().to(device))
			x = torch.from_numpy(label).long().to(device)
			loss = criteration(y, torch.from_numpy(label).long().to(device))
			#print('loss', loss)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			y = y.cpu().detach().numpy()
			train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]


		test_loss = []
		test_acc = []
		while True:
			with torch.no_grad():
				try:
					text, label = generator_test.__next__()
				except:
					break
				y = model(torch.from_numpy(text).long().to(device))
				loss = criteration(y, torch.from_numpy(label).long().to(device))
				test_loss.append(loss.item())
				y = y.cpu().numpy()
				test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'.format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))


		if np.mean(test_acc) > best_acc:
			best_acc = np.mean(test_acc)
		else:
			early_stop += 1  # 早停机制  0.566 0.533 ...0.110
			                 # 0.566 0.533 0.522 0.566 0.544
		if early_stop >= STOP:
			logger.info('best acc: {:.4f}'.format(best_acc))
			return best_acc

		# if (epoch + 1) % SAVE_EVERY == 0:
# 			print('saving parameters')
# 			os.makedirs('models', exist_ok=True)
# 			torch.save(model.state_dict(), 'models/textrnn-' + str(epoch) + '.pkl')
	logger.info('best acc: {:.4f}'.format(best_acc))
	return best_acc


if __name__ == '__main__':
	acc = []

	#with open('./data/data_half/{}.txt'.format(args.filename_pos), 'r', encoding='utf-8') as f:
	with open(r'D:\Destop\1\bal=0.2-50_5k.txt', 'r', encoding='utf-8') as f:  # 有空格
		raw_pos = f.read().split("\n")
	# filter第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False,最后将返回 True 的元素放到新列表中。
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	#raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]

	#with open('./data/data_half/{}.txt'.format(args.filename_neg), 'r', encoding='utf-8') as f:
	with open(r'D:\Destop\1\ori_two_50.txt', 'r', encoding='utf-8') as f:   # 也有空格,把40的复制进去
		raw_neg = f.read().split("\n")
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	#raw_neg = [' '.join(list(jieba.cut(pp))) for pp in raw_neg]

	data_helper = data.DataHelper([raw_pos, raw_neg], use_label=True)
	for i in range(1):
		acc.append(main(data_helper))
	acc_mean = np.mean(acc)
	logger.info("best acc : {:.4f}".format(min(acc)))
	logger.info("worst acc: {:.4f}".format(max(acc)))
	logger.info("acc final: {:.4f}+{:.4f}".format(acc_mean, max(acc_mean - min(acc), max(acc) - acc_mean)))

