import collections
import numpy as np

def pxy(line,sigle,two_base):
	line = line[1:len(line)-1]
	temp = ''
	for i in range(0,len(line)-1,2):
		temp += line[i] + line[i+1] + ' '

	list_temp= temp.split(' ')
	temp_ = collections.Counter(list_temp)
	out = {}
	dict_single = {}
	dict_two = {}
	len_all = 0
	for base,value in two_base:
		dict_two[base] = value + temp_[base]
		len_all += value

	for base,value in sigle:
		dict_single[base] = value
	len_all = len_all *2


	for base, value in two_base:
		first_num = dict_single[base[0]]
		second_num = dict_single[base[1]]
		out[base] = (dict_two[base] * len_all) / (first_num * second_num)

	out = sorted(collections.Counter(out).items(),key= lambda x : x[1],reverse=True)

	return out


class Vocabulary_pxy(object):
	def __init__(self, data_path, max_len=200, min_len=5, word_drop=100, encoding='utf8'):
		if type(data_path) == str:
			data_path = [data_path]
		self._data_path = data_path
		self._max_len = max_len
		self._min_len = min_len
		#self._word_drop = word_drop
		self._word_drop = 0
		self._encoding = encoding
		self.token_num = 0
		self.vocab_size_raw = 0
		self.vocab_size = 0
		self.w2i = {}
		self.i2w = {}
		self.start_words = []
		self._build_vocabulary()

	def _build_vocabulary(self):
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'
		words_all = []
		start_words = []
		for data_path in self._data_path:
			with open(data_path, 'r', encoding=self._encoding) as f:
				sentences = f.readlines()
			for sentence in sentences:
				_ = sentence.split()
				if (len(_) >= self._min_len) and (len(_) <= self._max_len):
					words_all.extend(_)
					start_words.append(_[0])
		self.token_num = len(words_all)

		str_temp = list(''.join(words_all))
		singlebase = sorted(collections.Counter(str_temp).items(),key=lambda  x:x[1],reverse=True)


		word_distribution = sorted(collections.Counter(words_all).items(), key=lambda x: x[1], reverse=True)#获得文件中各个单词的分布

		word_distribution = pxy(str_temp,singlebase,word_distribution)

		self.vocab_size_raw = len(word_distribution)
		for (word, value) in word_distribution:
			if value > self._word_drop:#丢弃出现概率过小的词语
				self.w2i[word] = len(self.w2i) #筛选过后将word与对应的频数传递给w2i数组
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)
		start_word_distribution = sorted(collections.Counter(start_words).items(), key=lambda x: x[1], reverse=True)#获取序列起始单词的分布
		self.start_words = [_[0] for _ in start_word_distribution]


class Vocabulary(object):
	def __init__(self, data_path, max_len=200, min_len=5, word_drop=5, encoding='utf8'):
		if type(data_path) == str:
			data_path = [data_path]
		self._data_path = data_path
		self._max_len = max_len
		self._min_len = min_len
		self._word_drop = word_drop
		self._encoding = encoding
		self.token_num = 0
		self.vocab_size_raw = 0
		self.vocab_size = 0
		self.w2i = {}
		self.i2w = {}
		self.start_words = []
		self._build_vocabulary()

	def _build_vocabulary(self):
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'
		words_all = []
		start_words = []
		for data_path in self._data_path:
			with open(data_path, 'r', encoding=self._encoding) as f:
				sentences = f.readlines()
			for sentence in sentences:
				_ = sentence.split()
				if (len(_) >= self._min_len) and (len(_) <= self._max_len):
					words_all.extend(_)
					start_words.append(_[0])
		self.token_num = len(words_all)

		word_distribution = sorted(collections.Counter(words_all).items(), key=lambda x: x[1],reverse=True)  # 获得文件中各个单词的分布
		self.vocab_size_raw = len(word_distribution)
		for (word, value) in word_distribution:
			if value > self._word_drop:  # 丢弃出现概率过小的词语
				self.w2i[word] = len(self.w2i)  # 筛选过后将word与对应的频数传递给w2i数组
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)
		start_word_distribution = sorted(collections.Counter(start_words).items(), key=lambda x: x[1],
										 reverse=True)  # 获取序列起始单词的分布
		self.start_words = [_[0] for _ in start_word_distribution]


class Corpus(object):
	def __init__(self, data_path, vocabulary, max_len=200, min_len=5):
		if type(data_path) == str:
			data_path = [data_path]
		self._data_path = data_path
		self._vocabulary = vocabulary
		self._max_len = max_len
		self._min_len = min_len
		self.corpus = []
		self.corpus_length = []
		self.labels = []
		self.sentence_num = 0
		self.max_sentence_length = 0
		self.min_sentence_length = 0
		self._build_corpus()

	def _build_corpus(self):
		def _transfer(word):
			try:
				return self._vocabulary.w2i[word]
			except:
				return self._vocabulary.w2i['_UNK']
		label = -1
		for data_path in self._data_path:
			label += 1
			with open(data_path, 'r', encoding='utf8') as f:
				sentences = f.readlines()
			for sentence in sentences:
				sentence = sentence.split()
				if (len(sentence) >= self._min_len) and (len(sentence) <= self._max_len):
					sentence = ['_BOS'] + sentence + ['_EOS']   #给语句加上开始与结束标识
					self.corpus.append(list(map(_transfer, sentence)))  #将词语word映射为数字序列
					self.labels.append(label)      #每有一句sentence，将标签序列+1
		self.corpus_length = [len(i) for i in self.corpus]  #length为语句的个数
		self.max_sentence_length = max(self.corpus_length)  #最长为102 = 100（碱基数据） + 2（开头与末尾标识）
		self.min_sentence_length = min(self.corpus_length)  #最短也为102，同上
		self.sentence_num = len(self.corpus)   # sentence_num 为语句的个数


def split_corpus(data_path, train_path, test_path, max_len=200, min_len=5, ratio=0.8, seed=0, encoding='utf8'):
	with open(data_path, 'r', encoding=encoding) as f:
		sentences = f.readlines()
	sentences = [_ for _ in filter(lambda x: x not in [None, ''], sentences)  #剔除空数据与None
	             if len(_.split()) <= max_len and len(_.split()) >= min_len]
	np.random.seed(seed)
	np.random.shuffle(sentences) #打乱序列中的word，随机排序
	train = sentences[:int(len(sentences) * ratio)]  #train取sentence的前0.9
	test = sentences[int(len(sentences) * ratio):]   #test取sentence的后0.1 //ratio = 0.9
	with open(train_path, 'w', encoding='utf8') as f:
		for sentence in train:
			f.write(sentence)                 #写入train
	with open(test_path, 'w', encoding='utf8') as f:
		for sentence in test:
			f.write(sentence)                 #写入test


class Generator(object):
	def __init__(self, data, vocabulary=None):
		self._data = np.array(data)
		self._vocabulary = vocabulary

	def _padding(self, batch_data):
		assert self._vocabulary is not None
		max_length = max([len(i) for i in batch_data])   #max为数据行最长的那一条
		for i in range(len(batch_data)):
			batch_data[i] += [self._vocabulary.w2i["_PAD"]] * (max_length - len(batch_data[i]))  #零填充，让不等长的数据等长为32
			#batch_data[i] += [self._vocabulary.w2i["_PAD"]] * (max_length - 3)
		return np.array(list(batch_data))

	def build_generator(self, batch_size, shuffle=True, padding=False):
		indices = list(range(len(self._data)))   #返回data数据组的维度，对于train数据集而言，为9000。
		if shuffle:
			np.random.shuffle(indices)   #对数据列表随机排序
		while True:
			batch_indices = indices[0:batch_size]               # 产生一个batch的index
			indices = indices[batch_size:]                      # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = self._data[batch_indices]    #batch_size = 32，将原数据按行随机打乱，按32条为一组存储在batch_data中，
			if padding:
				batch_data = self._padding(batch_data)
			yield batch_data


# class Generator(object):
# 	def __init__(self, data):
# 		self._data = data
#
# 	def build_generator(self, batch_size, sequence_len, shuffle=True):
# 		if shuffle:
# 			np.random.shuffle(self._data)
# 		data_ = []
# 		for _ in self._data:
# 			data_.extend(_)
# 		batch_num = len(data_) // (batch_size * sequence_len)
# 		data = data_[:batch_size * batch_num * sequence_len]
# 		data = np.array(data).reshape(batch_num * batch_size, sequence_len)
# 		while True:
# 			batch_data = data[0:batch_size]                   # 产生一个batch的index
# 			data = data[batch_size:]                          # 去掉本次index
# 			if len(batch_data) == 0:
# 				return True
# 			yield batch_data
