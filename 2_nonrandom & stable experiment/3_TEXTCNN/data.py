import numpy as np
import collections


class DataHelper(object):
	def __init__(self, raw, word_drop=5, ratio=0.8, use_label=False, use_length=False):
		assert (use_label and (len(raw) == 2)) or ((not use_label) and (len(raw) == 1))
		self._word_drop = word_drop

		self.use_label = use_label
		self.use_length = use_length

		self.train = None
		self.train_num = 0
		self.test = None
		self.test_num = 0
		if self.use_label:
			self.label_train = None
			self.label_test = None
		if self.use_length:
			self.train_length = None
			self.test_length = None
			self.max_sentence_length = 0
			self.min_sentence_length = 0

		self.vocab_size = 0
		self.vocab_size_raw = 0
		self.sentence_num = 0
		self.word_num = 0

		self.w2i = {}
		self.i2w = {}

		sentences = []
		for _ in raw:
			sentences += _  # sentences是[10000行] + [10000行]

		self._build_vocabulary(sentences)
		corpus_length = None
		label = None
		if self.use_length:
			corpus, corpus_length = self._build_corpus(sentences)
		else:
			corpus = self._build_corpus(sentences)
		if self.use_label:
			label = self._build_label(raw)
		self._split(corpus, ratio, corpus_length=corpus_length, label=label)  # ratio = 0.8，即80%用于训练，20%用于测试

	def _build_label(self, raw):
		label = [0]*len(raw[0]) + [1]*len(raw[1])
		return np.array(label)

	def _build_vocabulary(self, sentences):
		self.sentence_num = len(sentences)
		words = []
		for sentence in sentences:
			words += sentence.strip().split(' ')  # 每一个碱基算一个word
		self.word_num = len(words)
		# collectionsd的Counter用于统计词频
		# sorted(, key= ), 以lambda x: x[1] ，用key的值来进行排序， 起始是出现的次数
		# 然后x[1]每个word出现的次数
		word_distribution = sorted(collections.Counter(words).items(), key=lambda x: x[1], reverse=True)
		self.vocab_size_raw = len(word_distribution)
		self.w2i['_PAD'] = 0  # word to index  A:0  T:1
		self.w2i['_UNK'] = 1  # UNK表示的是未知字符 unknown
		self.w2i['_BOS'] = 2  # begin of sentence
		self.w2i['_EOS'] = 3  # end of sentence
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'

		for (word, value) in word_distribution:  # word_distribution (A:, T:, C:, G:)
			if value > self._word_drop:  # value是出现的频率
				self.w2i[word] = len(self.w2i)
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)

	def _build_corpus(self, sentences):  # 这里返回的corpus维度(语料库就是20000,3行)
		def _transfer(word):
			try:
				return self.w2i[word]
			except:
				return self.w2i['_UNK']
		# map() 会根据提供的函数对指定序列做映射。
		# 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
		corpus = [[self.w2i["_BOS"]] + list(map(_transfer, sentence.split(' '))) + [self.w2i["_EOS"]] for sentence in sentences]
		if self.use_length:
			corpus_length = np.array([len(i) for i in corpus])
			self.max_sentence_length = corpus_length.max()
			self.min_sentence_length = corpus_length.min()
			return np.array(corpus), np.array(corpus_length)
		else:
			return np.array(corpus)

	def _split(self, corpus, ratio, corpus_length=None, label=None):
		indices = list(range(self.sentence_num))  # sentence_num = 20000
		np.random.shuffle(indices)
		self.train = corpus[indices[:int(self.sentence_num * ratio)]]
		self.train_num = len(self.train)
		self.test = corpus[indices[int(self.sentence_num * ratio):]]
		self.test_num = len(self.test)
		if self.use_length:  # use_length为false
			self.train_length = corpus_length[indices[:int(self.sentence_num * ratio)]]
			self.test_length = corpus_length[indices[int(self.sentence_num * ratio):]]
		if self.use_label:
			self.label_train = label[indices[:int(self.sentence_num * ratio)]]
			self.label_test = label[indices[int(self.sentence_num*ratio):]]

	def _padding(self, batch_data):
		max_length = max([len(i) for i in batch_data])
		for i in range(len(batch_data)):
			batch_data[i] += [self.w2i["_PAD"]] * (max_length - len(batch_data[i]))
		return np.array(list(batch_data))

	def train_generator(self, batch_size, shuffle=True):
		indices = list(range(self.train_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0:batch_size]                   # 产生一个batch的index
			indices = indices[batch_size:]                          # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = self.train[batch_indices]
			# batch_data = self._padding(batch_data)
			result = [batch_data]
			if self.use_length:
				batch_length = self.train_length[batch_indices]
				result.append(batch_length)
			if self.use_label:
				batch_label = self.label_train[batch_indices]
				result.append(batch_label)
			yield tuple(result)

	def test_generator(self, batch_size, shuffle=True):
		indices = list(range(self.test_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0:batch_size]                   # 产生一个batch的index
			indices = indices[batch_size:]                          # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = self.test[batch_indices]
			# batch_data = self._padding(batch_data)
			result = [batch_data]
			if self.use_length:
				batch_length = self.test_length[batch_indices]
				result.append(batch_length)
			if self.use_label:
				batch_label = self.label_test[batch_indices]
				result.append(batch_label)
			yield tuple(result)

	pass


if __name__ == '__main__':

	with open('../_data/rt-polaritydata/rt-polarity.pos', 'r', encoding='Windows-1252') as f:
		raw_pos = list(f.readlines())
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	with open('../_data/rt-polaritydata/rt-polarity.neg', 'r', encoding='Windows-1252') as f:
		raw_neg = list(f.readlines())
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	data_helper = DataHelper([raw_pos, raw_neg], use_length=True, use_label=True)
	generator = data_helper.train_generator(64)
	i = 0
	a, b, c = generator.__next__()
	while True:
		try:
			a, b, c = generator.__next__()
			i += 1
			print(i)
		except:
			break
	pass






