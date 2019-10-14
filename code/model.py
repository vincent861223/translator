import torch.nn as nn
import torch
class Encoder(nn.Module):
	def __init__(self, vocab_size, emb_size, hidden_size, rnn_cell='GRU', padding_idx=0, bidirectional=False, n_layers=1, dropout=0.2, device='cpu'):
		super(Encoder, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.rnn_cell = rnn_cell
		self.padding_idx = padding_idx
		self.bidirectional = bidirectional
		self.n_layers = n_layers
		self.dropout = dropout
		self.device = device
		self.n_init = (2 if bidirectional == True else 1) * n_layers

		self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
		if rnn_cell == 'GRU': self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, dropout=dropout)

	def forward(self, source):
		# source: (batch, seq_len)
		init_hidden = torch.randn(self.n_init, source.size(0), self.hidden_size).to(self.device) #(n_layer*n_direction, batch, hidden_size)
		source = self.embedding(source) # (batch, seq_len) -> (batch, seq_len, emb_size)
		output, hidden = self.rnn(source, init_hidden) #(batch, seq_len, emb_size) -> (batch, seq_len, emb_size*n_direction), (n_layer*n_direction, batch, hidden_size)
		return output, hidden #(n_layer*n_direction, batch, hidden_size)


class Decoder(nn.Module):
	def __init__(self, vocab_size, emb_size, hidden_size, rnn_cell='GRU', padding_idx=0, bidirectional=False, n_layers=1, dropout=0.2, device='cpu'):
		super(Decoder, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.rnn_cell = rnn_cell
		self.padding_idx = padding_idx
		self.bidirectional = bidirectional
		self.n_layers = n_layers
		self.dropout = dropout
		self.device = device
		self.n_init = (2 if bidirectional == True else 1) * n_layers

		self.relu = nn.ReLU()
		self.softmax = nn.LogSoftmax(dim=-1)
		self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
		self.linear = nn.Linear(hidden_size, vocab_size)
		if rnn_cell == 'GRU': self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, dropout=0.2)

	def forward(self, input, init_hidden):
		# source: (batch, seq_len)
		input = self.relu(self.embedding(input)) # (batch, seq_len) -> (batch, seq_len, emb_size)
		output, hidden = self.rnn(input, init_hidden) #(batch, seq_len, emb_size) -> (batch, seq_len, emb_size*n_direction), (batch, n_layer*n_direction, hidden_size)
		output = self.softmax(self.linear(output))
		return output









