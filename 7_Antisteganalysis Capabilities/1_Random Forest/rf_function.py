from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import cg_tm_kl
import data
from gensim.models.word2vec import Word2Vec
import multiprocessing
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import csv
import os
import logger
from logger import Logger
def split_words(line,num):
	words = []
	for i in range(len(line)):
		words.append(line[i:i+num])

	return words

def total_vector(words,word2vec):
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += word2vec.wv[word].reshape((1, 300))
        except KeyError:
            continue
    return vec

def randomforest(x_train, y_train):
	classifier = RandomForestClassifier(n_estimators=200).fit(x_train, y_train)
	test_accuracy = classifier.score(x_train, y_train)
	joblib.dump(classifier, 'rf.pkl')
	return test_accuracy

def train_svm(x_train, y_train):
	svc = svm.SVC(verbose=True)
	parameters = {'C': [1, 2], 'gamma': [0.5, 1, 2]}
	clf = GridSearchCV(svc, parameters, scoring='f1')
	clf.fit(x_train, y_train, )
	print('最佳参数: ')
	print(clf.best_params_)

	# clf = svm.SVC(kernel='rbf', C=2, gamma=2, verbose=True)
	# clf.fit(x_train,y_train)

	# 封装模型
	print('保存模型...')
	joblib.dump(clf, 'svm.pkl')

def SVM(path_sc,path_ori):
	# pathsc = r'D:\Destop\seqs\660kb_3300\660_3300_output\read_5\ari\fxy5-50_100.txt'
	pathsc = path_sc
	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=200, beg_sc=0, end_sc=3300, PADDING=False, flex=10, num1=200)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	# raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]
	pathwrite_sc = r'D:\Destop\seqs\660kb_3300\660_3300_output\read_5\ari\1.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')

	# with open('./data/data_half/{}.txt'.format(args.filename_neg), 'r', encoding='utf-8') as f:
	# pathori = r'D:\Destop\seqs\660kb_3300\read_in_5\read_5.txt'
	pathori = path_ori
	raw_neg = cg_tm_kl.txt_process_sc_duo(pathori, len_sc=200, beg_sc=0, end_sc=3300, PADDING=False, flex=10, num1=200)
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

	pathwrite_ori = r'D:\Destop\seqs\660kb_3300\660_3300_output\read_5\adg\read_5_split1.txt'
	with open(pathwrite_ori, 'w') as f2:
		for line in raw_neg:
			f2.write(line)
			f2.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)
	neg = pd.read_csv(pathwrite_ori, header=None)

	pos['words'] = pos[0].apply(lambda x: split_words(x, 5))
	neg['words'] = neg[0].apply(lambda x: split_words(x, 5))

	x = np.concatenate((pos['words'], neg['words']))
	y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
	save = pd.DataFrame(x, y)
	# save.to_csv('data_3300.csv')

	word2vec = Word2Vec(x, vector_size=300, window=3, min_count=5, sg=1, hs=1, epochs=10, workers=25)
	#word2vec.save('word2vec.model')

	train_vec = np.concatenate([total_vector(words,word2vec) for words in x])

	x_train, x_test, y_train, y_test = train_test_split(train_vec, y, test_size=0.3, random_state=5)

	train_svm(x_train, y_train)

	svm = joblib.load('svm.pkl')

	y_pred = svm.predict(x_test)

	logger.info('SVM')
	logger.info(pathsc)
	logger.info(classification_report(y_test, y_pred))

def rf(path_sc,path_ori,lensc,ind):

	pathsc = path_sc

	with open(path_ori , 'r') as f1:
		lines = f1.readlines()
	LEN = len(lines)

	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=lensc, beg_sc=0, end_sc=LEN, PADDING=False, flex=10, num1=lensc)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

	pathwrite_sc = r'D:\Desktop\additional data\1.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')


	pathori = path_ori
	raw_neg = cg_tm_kl.txt_process_sc_duo(pathori, len_sc=lensc, beg_sc=0, end_sc=LEN, PADDING=False, flex=10, num1=lensc)
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

	pathwrite_ori = r'D:\Desktop\additional data\2.txt'
	with open(pathwrite_ori, 'w') as f2:
		for line in raw_neg:
			f2.write(line)
			f2.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)
	neg = pd.read_csv(pathwrite_ori, header=None)

	pos['words'] = pos[0].apply(lambda x: split_words(x, int(ind)))
	neg['words'] = neg[0].apply(lambda x: split_words(x, int(ind)))

	x = np.concatenate((pos['words'], neg['words']))
	y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
	save = pd.DataFrame(x, y)
	# save.to_csv('data_3300.csv')

	word2vec = Word2Vec(x, vector_size=300, window=3, min_count=5, sg=1, hs=1, epochs=10, workers=25)
	#word2vec.save('word2vec.model')

	train_vec = np.concatenate([total_vector(words,word2vec) for words in x])

	x_train, x_test, y_train, y_test = train_test_split(train_vec, y, test_size=0.2, random_state=5)

	randomforest(x_train, y_train)

	rf = joblib.load('rf.pkl')

	y_pred = rf.predict(x_test)

	#logger.info('RF')
	#logger.info(pathsc)
	tt = classification_report(y_test, y_pred,digits=4)
	for element in tt.split('\\n'):
		if element.find('accuracy') > 0:
			for ele in  element.split():
				try:
					if float(ele) < 1:
						acc = float(ele)
				except:
					continue

	#logger.info(classification_report(y_test, y_pred,digits=4))

	return acc

def csv_shengc(pathsc):
	dirs_name = r'D:\Destop\seqs\BlanExperiments\CP2\880NormalSplit\CSV'
	# pathsc = r'D:\Destop\seqs\traditionnal method\fxy_LSB_METHOD.txt'
	# pathsc = path_sc
	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=200, beg_sc=0, end_sc=2250, PADDING=False, flex=10, num1=200)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	# raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]
	pathwrite_sc = r'D:\Destop\seqs\660kb_3300\660_3300_output\read_5\ari\1.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')

	# with open('./data/data_half/{}.txt'.format(args.filename_neg), 'r', encoding='utf-8') as f:
	# pathori = r'D:\Destop\seqs\660kb_3300\read_in_5\read_5.txt'
	# pathori = path_ori
	raw_neg = cg_tm_kl.txt_process_sc_duo(pathori, len_sc=200, beg_sc=0, end_sc=2250, PADDING=False, flex=10, num1=200)
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

	pathwrite_ori = r'D:\Destop\seqs\660kb_3300\660_3300_output\read_5\adg\read_5_split1.txt'
	with open(pathwrite_ori, 'w') as f2:
		for line in raw_neg:
			f2.write(line)
			f2.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)
	neg = pd.read_csv(pathwrite_ori, header=None)

	# pos['words'] = pos[0].apply(lambda x: split_words(x, 5))
	# neg['words'] = neg[0].apply(lambda x: split_words(x, 5))

	x = np.concatenate((pos, neg))
	y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

	save = pd.DataFrame(columns=['text', 'label'], index=range(4500))
	# save = pd.DataFrame(x,y)
	save['text'] = x
	save['label'] = y


	t = pathsc[pathsc.rfind('\\')  + 1 : len(pathsc)-4]

	#file_ = r'C:\Users\shmily\PycharmProjects\SVM\csv/' + pathsc[pathsc.find('traditionnal method') +len('traditionnal method') + 1 : len(pathsc)-4] + '.csv'

	file_ = dirs_name + t + '.csv'

	save.to_csv(file_)


if __name__ == '__main__':
	FILE_mode = 'b' #"b":baselin mode // "a":ADG mode

	mode = '43'

	if FILE_mode == 'a':
		index = ['2','3','4','5','6','8','10','20','33','40','50','66']

		file_finaly = r'D:\Desktop\additional data\{}\RF\{}_ADG.csv'.format(mode, mode)  # 所有文件的得分值集合

	else:
		index = ['4'] #在优选的SL下生成baselines，二分类任务时只需要固定一个index即可

		file_finaly = r'D:\Desktop\additional data\{}\RF\{}_baselines.csv'.format(mode, mode)  # 所有文件的得分值集合

	All = pd.DataFrame(columns=['path', 'accurcy'])

	file_name = ''
	if mode == '18':
		file_name = r'ASM1821919v1'
	elif mode == '28':
		file_name = r'ASM286374v1'
	else:
		file_name = r'GCA_004358755'

	for ind in index:
		temp = pd.DataFrame(columns=['path', 'accurcy'])
		if FILE_mode == 'a':
			log_file = r"D:\Desktop\additional data\{}\RF\RF_ADG_log.txt".format(mode) #过程记录

			file_generated = r'D:\Desktop\additional data\{}\seq\{}'.format(mode, ind) #生成文件的存放文件夹位置
		else:
			log_file = r"D:\Desktop\additional data\{}\RF\RF_baselines_log.txt".format(mode)

			file_generated = r'D:\Desktop\additional data\{}\baselines'.format(mode)

		logger = Logger(log_file)
		PATH_generated = []
		BPN = {}

		if int(ind) % 3 == 0:
			SeqLength = 198
			file_original = r'D:\Desktop\additional data\{}\base_extaction\{}_198_{}.txt'.format(mode, file_name, ind)
		else:
			SeqLength = 200
			file_original = r'D:\Desktop\additional data\{}\base_extaction\{}_200_{}.txt'.format(mode, file_name, ind)

		with open(file_original, 'r') as f1:
			lines = f1.readlines()

		for root, dirs, files in os.walk(file_generated):
			for file in files:
				PATH_generated.append(os.path.join(root, file))

		for file_g in PATH_generated:
			p_ = file_g[file_g.rfind('\\') + 1:  len(file_g) - 4]
			BPN[p_] = cg_tm_kl.find_bpn(file_g)
			accurcy = rf(file_g, file_original, endsc=len(lines), lensc=SeqLength,ind =ind)
			temp = temp.append([{'path': p_, 'accurcy': accurcy}], ignore_index=True)
			All = All.append([{'path': p_, 'accurcy': accurcy}], ignore_index=True)

		accurcy_average = np.mean(pd.to_numeric(temp['accurcy']).tolist())

		accurcy_up = max(pd.to_numeric(temp['accurcy']).tolist()) - accurcy_average

		accurcy_down = accurcy_average - min(pd.to_numeric(temp['accurcy']).tolist())

		accurcy_std = np.std((pd.to_numeric(temp['accurcy']).tolist()))

		All = All.append([{'path': 'ave', 'accurcy': accurcy_average}], ignore_index=True)

		All = All.append([{'path': 'ACC_STD', 'accurcy': accurcy_std}], ignore_index=True)

		All = All.append([{'path': 'Pos ERR', 'accurcy': accurcy_up}], ignore_index=True)

		All = All.append([{'path': 'Neg ERR', 'accurcy': accurcy_down}], ignore_index=True)

	All.to_csv(file_finaly)







