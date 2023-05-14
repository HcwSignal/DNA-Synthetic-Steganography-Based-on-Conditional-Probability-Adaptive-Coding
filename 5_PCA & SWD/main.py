import random
import os
import sys

from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import cg_tm_kl
#import data
from gensim.models.word2vec import Word2Vec
import multiprocessing
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import csv
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
colors = ['blue','orange']#设置散点颜色
maker=['.','.']#设置散点形状
from scipy.stats import wasserstein_distance
import EMD
import random
import pathlib
import mpl_toolkits.axisartist as axisartist
def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]
  s_ = ytree.query(x, k=1, eps=.01, p=2)
  r_ = xtree.query(x, k=1, eps=.01, p=2)[0]
  r_soreted = sorted(r,reverse=True)
  s_soreted = sorted(s, reverse=False)
  for i in range(len(r)):
  	if abs(r[i]) < 0.0001: r[i] = 0.0001

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def plot_with_labels(S_lowDWeights,Trure_labels,path):
	plt.cla()  # 清除当前图形中的当前活动轴,所以可以重复利用
	# 创建画布
	#fig = plt.figure(figsize=(12, 12))
	# 使用axisartist.Subplot方法创建一个绘图区对象ax
	#ax = axisartist.Subplot(fig, 111)
	# 将绘图区对象添加到画布中
	# fig.add_axes(ax)
	fig = plt.figure()
	ax = axisartist.Subplot(fig, 111)
	fig.add_subplot(ax)
	# 通过set_visible方法设置绘图区所有坐标轴隐藏
	ax.axis["right"].set_visible(False)
	ax.axis["top"].set_visible(False)


	# 降到二维了，分别给x和y
	True_labels = Trure_labels.reshape((-1, 1))

	S_data = np.hstack((S_lowDWeights, True_labels))
	S_data = pd.DataFrame({
		'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

	for index in range(2):
		X = S_data.loc[S_data['label'] == index]['x']
		Y = S_data.loc[S_data['label'] == index]['y']
		plt.scatter(X, Y,marker=maker[index],color= colors[index], alpha=0.65)

	#plt.xticks([])  # 去掉横坐标值
	#plt.yticks([])  # 去掉纵坐标值
	#
	#plt.title(name, fontsize=32, fontweight='normal', pad=20)
	#plt.xlabel('Original DNA')
	#plt.ylabel('Stega DNA')

	# my_x_ticks = np.arange(-40, 70, 10)
	# my_y_ticks = np.arange(-40, 70, 10)
	my_x_ticks = np.arange(-8, 12, 4)
	my_y_ticks = np.arange(-8, 12, 4)
	plt.xticks(my_x_ticks)
	plt.yticks(my_y_ticks)
	# plt.xticks([])
	# plt.yticks([])

	# plt.legend(['Original DNA','Stega DNA'])
	savename  = path[path.rfind('\\') + 1:len(path)-4]
	plt.savefig('new\\' + savename + '.jpg')
	#plt.show()

def split_words(line,num):
	words = []
	for i in range(0,len(line),num):
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


def yangbenqianru(pathsc,word2vec_emd,ind,Path_save):
	if int(ind) % 3 == 0: ##序列只存在两种情况（198、200）
		len_sc = 198
	else:
		len_sc = 200

	split_words_ = int(ind)

	with open(pathsc,'r') as f1:
		lines = f1.readlines()

	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=len_sc, beg_sc=0, end_sc=len(lines), PADDING=False, flex=10, num1=len_sc,
										  tiqu=False)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

	pathwrite_sc = Path_save + r'\1.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)

	pos['words'] = pos[0].apply(lambda x: split_words(x, split_words_)) #####需要根据不同的样本修改

	t = pos['words']

	t_vec = [total_vector(words, word2vec_emd) for words in t]

	return t_vec,len_sc,split_words_


def yangbenqianru_SC(pathsc, word2vec_emd,LEN_SC,SPLIT_NUM_,Path_save):

	with open(pathsc, 'r',encoding='gbk') as f1:
		lines = f1.readlines()

	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=LEN_SC, beg_sc=0, end_sc=len(lines), PADDING=False, flex=10,
										  num1=LEN_SC,
										  tiqu=False)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

	pathwrite_sc = Path_save + r'\2.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)

	pos['words'] = pos[0].apply(lambda x: split_words(x, SPLIT_NUM_))  #####需要根据不同的样本修改

	t = pos['words']

	t_vec = [total_vector(words, word2vec_emd) for words in t]

	return t_vec

def yangbenqianru_word(pathsc):
	with open(pathsc, 'r',encoding='gbk') as f1:
		lines = f1.readlines()

	raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=198, beg_sc=0, end_sc=len(lines), PADDING=False, flex=10, num1=198,
										  tiqu=False)
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

	pathwrite_sc = r'D:\Destop\seqs\traditionnal method\1.txt'
	with open(pathwrite_sc, 'w') as f1:
		for line in raw_pos:
			f1.write(line)
			f1.write('\n')

	pos = pd.read_csv(pathwrite_sc, header=None)

	pos['words'] = pos[0].apply(lambda x: split_words(x, 3))

	t = pos['words']


	return t

if __name__ == '__main__':
	filemode = 'x' #"x":xiaorong_shiyan

	mode_file_name = {}
	mode_file_name['160'] = r'ena_178901_noncoding'
	mode_file_name['221'] = r'ena_1909293_noncoding'
	mode_file_name['572'] = r'ena_438_noncoding'
	mode_file_name['611'] = r'ena_39491_noncoding'
	mode_file_name['43']  = r'GCA_004358755'
	mode_file_name['660'] = r'read'

	modes = ['572']


	#ind = '4'

	for mode in modes:
		if mode == '660':
			ind = '3'
			Model_yinshe = 'OriginalFile=888_read_3.model'
		else:
			ind = '4'
			Model_yinshe = 'OriginalFile=888_read_4.model'

		Path_result_save = r'D:\Desktop\PaperCode_2023\ExperimentData\{}'.format(mode)

		if int(ind) % 3 == 0:
			SeqLength = 198
			OriginalFile = Path_result_save + r'\base_extaction\{}_198_{}.txt'.format(str(mode_file_name[mode]), ind)
		else:
			SeqLength = 200
			OriginalFile = Path_result_save + r'\base_extaction\{}_200_{}.txt'.format(str(mode_file_name[mode]), ind)

		with open(OriginalFile, 'r') as f1:
			lines = f1.readlines()

		number_sentence = len(lines)

		model_load = True

		random.seed(50)

		np.random.seed(50)

		pathsc = r'D:\Destop\file\科研相关\论文代码\ExperimentData\847\read_4.txt'

		pathori = OriginalFile

		dirs = Path_result_save + r'\baselines\xiaorong_experiment'.format(mode)

		pathlib.Path(dirs).mkdir(parents=True, exist_ok=True)

		#Model_yinshe = 'OriginalFile=888_read_3.model'

		write_dirs = Path_result_save + r'\PCA'

		pathlib.Path(write_dirs).mkdir(parents=True, exist_ok=True)

		P = []

		for root, dirs, files in os.walk(dirs):
			for file in files:
				P.append(os.path.join(root, file))

		if model_load:
			word2vec_emd = Word2Vec.load(Model_yinshe)  # 在一个无关的样本上做映射
		# word2vec_emd = Word2Vec.load('ori.model')  # 在原样本上进行映射
		else:
			# 在两个样本合并后做映射
			tx_word = yangbenqianru_word(pathsc)
			ty_word = yangbenqianru_word(pathori)
			x = np.concatenate((tx_word, ty_word))
			word2vec_emd = Word2Vec(ty_word, vector_size=300, window=3, min_count=5, sg=1, hs=1, epochs=10, workers=25)
			word2vec_emd.save('OriginalFile=888_read_3.model')
			sys.exit()

		ty_, len_sc, split_words_ = yangbenqianru(pathori, word2vec_emd, ind, Path_result_save)  # 原始DNA高维向量表示

		ty_ = np.squeeze(ty_)

		embedded_y = PCA(n_components=2).fit_transform(ty_)

		# labels = np.ones(len(ty_))
		# embedded_y = embedded_y[: int(len(embedded_y) / 330), :]
		# plot_with_labels(embedded_y, labels, p)

		pd_EMD = pd.DataFrame(columns={'name', 'emd'})
		for p in P:

			embedded_x0 = []
			embedded_x1 = []
			embedded_y0 = []
			embedded_y1 = []

			tx_ = yangbenqianru_SC(p, word2vec_emd, len_sc, split_words_, Path_result_save)

			tx_ = np.squeeze(tx_)

			# train_vec = np.concatenate((tx_, ty_), axis=0)
			# train_vec = np.squeeze(np.array(train_vec))

			# embedded = TSNE(n_components=2, perplexity=30).fit_transform(train_vec) #TSNE

			embedded_x = PCA(n_components=2).fit_transform(tx_)  # PCA

			# embedded_x = [embedded_x[i] for i in random.sample(range(0,len(embedded_x)),int(len(embedded_x) / 3))]
			# embedded_x = embedded_x[ : int(len(embedded_x) / 330), : ]

			# num_embedded_y = embedded_y.shape[0]
			# num_embedded_x = embedded_x.shape[0]

			# EMD = emd1.doRubnerComparisonExample(embedded_x,embedded_y,num_embedded_x,num_embedded_y)

			embedded_x = embedded_x[:number_sentence - 10]
			embedded_y = embedded_y[:number_sentence - 10]
			for data in embedded_x:
				embedded_x0.append(data[0])
				embedded_x1.append(data[1])

			for data in embedded_y:
				embedded_y0.append(data[0])
				embedded_y1.append(data[1])

			Data = pd.DataFrame(columns={'sc_x', 'sc_y', 'ori_x', 'ori_y'})

			for i in range(min(len(embedded_x), len(embedded_y))):
				Data = Data.append(
					{'sc_x': embedded_x0[i], 'sc_y': embedded_x1[i], 'ori_x': embedded_y0[i], 'ori_y': embedded_y1[i]},
					ignore_index=True)

			emd = EMD.getEMD(embedded_x,embedded_y,len(embedded_x),p)

			#emd = 0

			Data = Data.append({'sc_x': '1', 'sc_y': '1', 'ori_x': '1', 'ori_y': emd},
							   ignore_index=True)

			p_name = p[p.rfind('\\') + 1: -4]

			writename = write_dirs + p_name + '.csv'

			Data.to_csv(writename, index=False, sep=',')

			p_name = p[p.rfind('\\') + 1:]
			pd_EMD = pd_EMD.append({'name': p_name, 'emd': emd}, ignore_index=True)

			print('finish!')

		# label = np.concatenate((np.ones(len(embedded_x)), np.zeros(len(embedded_y))), axis=0)
		# labels = label

		# embedded = np.concatenate((embedded_x,embedded_y),axis=0)

		# plot_with_labels(embedded, labels, p)

		print(pd_EMD)

		with open(Path_result_save + r'\baselines\代码参数({}).txt'.format(mode), 'w') as f1:
			f1.write(r'可视化坐标集合文件夹（绝对路径）：{}'.format(write_dirs))
			f1.write('\n')
			# f1.write(r'降维可视化映射文件：C:\Users\Administrator\PycharmProjects\emd\Original\888\read_3.txt')
			# f1.write('\n')
			f1.write(r'映射模型：{}'.format(Model_yinshe))

		pd_EMD.to_csv(Path_result_save + r'\baselines\{}_all_result_xiaorong.csv'.format(mode))

