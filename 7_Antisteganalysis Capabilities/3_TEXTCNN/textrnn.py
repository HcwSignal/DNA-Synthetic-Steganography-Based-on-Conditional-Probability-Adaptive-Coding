import torch
from torch import nn


class TextRNN(nn.Module):
	def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, class_num, dropout_rate):
		super(TextRNN, self).__init__()
		self._cell = cell

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.rnn = None
		if cell == 'rnn':
			self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = hidden_dim * num_layers
		elif cell == 'bi-rnn':
			self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'gru':
			self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = hidden_dim * num_layers
		elif cell == 'bi-gru':
			self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'lstm':
			self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
			out_hidden_dim = 2 * hidden_dim * num_layers
		elif cell == 'bi-lstm':
			self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			out_hidden_dim = 4 * hidden_dim * num_layers
		else:
			raise Exception("no such rnn cell")

		self.output_layer = nn.Linear(out_hidden_dim, class_num)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		"""
		:param x:(N,L)
		:return: (N,class_num)
		"""
		x = x.long()
		_ = self.embedding(x)
		_ = _.permute(1, 0, 2)
		__, h_out = self.rnn(_)
		if self._cell in ["lstm", "bi-lstm"]:
			h_out = torch.cat([h_out[0], h_out[1]], dim=2)
		h_out = h_out.permute(1, 0, 2)
		h_out = h_out.reshape(-1, h_out.shape[1]*h_out.shape[2])
		_ = self.output_layer(h_out)
		_ = self.softmax(_)
		return _


if __name__ == '__main__':
	"""# m = nn.AdaptiveMaxPool1d(5)
	# input = torch.randn(1, 64, 8)
	# output = m(input)
	textCnn = TextCNN(10,30,5,[3,4,5],2)
	x = torch.randint(low=0, high=10, size=(64,20))
	y = textCnn(x)
	criteration = nn.CrossEntropyLoss()
	optimizer = optim.SGD(textCnn.parameters(), lr=0.001)"""
	pass