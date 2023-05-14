import torch
from torch import nn
import torch.optim as optim
#import jieba
import argparse
import sys
from logger import Logger
# from torchsummary import summary

import numpy as np

import data
import textcnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x : x.lower() == 'true')
    parser.add_argument("--filename", type=str, default='20_ORI')
   # parser.add_argument("--filename_pos", type=str, default='ori_two')  # 正例子路径
    parser.add_argument("--filename_neg", type=str, default='20%')  # 反例子路径
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--stop", type=int, default=200)
    args = parser.parse_args(sys.argv[1:])
    return args
args = get_args()
# logger
log_file = "./log/cnn_two_5k/cnn_{}_ORI.txt".format(args.filename)
logger = Logger(log_file)

def main(data_helper):
	# ======================
	# 超参数
	# ======================
	BATCH_SIZE = 50
	EMBED_SIZE = 128
	FILTER_NUM = 128
	FILTER_SIZE = [3, 4, 5]
	CLASS_NUM = 2
	DROPOUT_RATE = 0.2
	EPOCH = args.epoch
	LEARNING_RATE = 0.001
	SAVE_EVERY = 20
	STOP = args.stop

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			logger.info("{0:15} ".format(var))
			logger.info(all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = textcnn.TextCNN(
		vocab_size=data_helper.vocab_size,
		embed_size=EMBED_SIZE,
		filter_num=FILTER_NUM,
		filter_size=FILTER_SIZE,
		class_num=CLASS_NUM,
		dropout_rate=DROPOUT_RATE
	)
	model.to(device)
# 	summary(model, (20,))
	criteration = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	early_stop = 0
	best_acc = 0 

	# ======================
	# 训练与测试
	# ======================
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)
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
			loss = criteration(y, torch.from_numpy(label).long().to(device))
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

		logger.info("epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}"
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
		      
		if np.mean(test_acc) > best_acc:
			best_acc = np.mean(test_acc)
		else:
			early_stop += 1
		if early_stop >= STOP:
			logger.info('best acc: {:.4f}'.format(best_acc))
			return best_acc
			
		# if (epoch + 1) % SAVE_EVERY == 0:
# 			print('saving parameters')
# 			os.makedirs('models', exist_ok=True)
# 			torch.save(model.state_dict(), 'models/textcnn-' + str(epoch) + '.pkl')
	logger.info('best acc: {:.4f}'.format(best_acc))
	return best_acc
		


if __name__ == '__main__':
	acc = []
	with open('data/data_half/ori_two.txt'.format(args.filename), 'r', encoding='utf-8') as f:
		raw_pos = f.read().split("\n")

	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	#raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]

	with open(('./data/data_half/ori_{}_two.txt'.format(args.filename_neg)), 'r', encoding='utf-8') as f:
		raw_neg = f.read().split("\n")
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	#raw_neg = [' '.join(list(jieba.cut(pp))) for pp in raw_neg]
	data_helper = data.DataHelper([raw_pos, raw_neg], use_label=True)
	for i in range(2):
		acc.append(main(data_helper))
	acc_mean = np.mean(acc)
	logger.info("best acc : {:.4f}".format(min(acc)))
	logger.info("worst acc: {:.4f}".format(max(acc)))
	logger.info("acc final: {:.4f}+{:.4f}".format(acc_mean, max(acc_mean - min(acc), max(acc) - acc_mean)))
