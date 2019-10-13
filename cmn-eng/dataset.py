from torch.utils.data import Dataset 

from utils import read_datafile
from lang import Lang
import torch


class SentencePairDataset(Dataset):
	def __init__(self, data_file, lang1_name, lang2_name, mode):
		self.lang1_name = lang1_name
		self.lang2_name = lang2_name
		self.mode = mode
		self.pairs = read_datafile(data_file, mode=mode)
		lang1_sents = [pair[0] for pair in self.pairs]
		lang2_sents = [pair[1] for pair in self.pairs]
		self.max_len_l1 = max(map(len, lang1_sents, ))
		self.max_len_l2 = max(map(len, lang2_sents))
		#self.max_len = max(self.max_len_l1, self.max_len_l2)
		self.max_len = 10
		self.lang1 = Lang(lang1_name, lang1_sents)
		self.lang2 = Lang(lang2_name, lang2_sents)
	
	def __str__(self):
		return 'lang1_name: {}\n'.format(self.lang1_name) + \
				'lang2_name: {}\n'.format(self.lang2_name) + \
				'max_len_l1: {}\n'.format(self.max_len_l1) + \
			   'max_len_l2: {}\n'.format(self.max_len_l2) + \
			   'max_len: {}\n'.format(self.max_len) + \
			   'n_pairs: {}\n'.format(self.__len__())

	def __getitem__(self, index):
		sent1, sent2 = self.pairs[index]
		indexed_sent1 = [self.lang1.word2index[w] for w in sent1]
		indexed_sent2 = [self.lang2.word2index[w] for w in sent2]
		indexed_sent1 = self.truncate_or_pad(indexed_sent1, bos=True, eos=True)
		indexed_sent2_bos = self.truncate_or_pad(indexed_sent2, bos=True)
		indexed_sent2_eos = self.truncate_or_pad(indexed_sent2, eos=True)
		return {'org': [sent1, sent2], 'indexed': [indexed_sent1, indexed_sent2_bos, indexed_sent2_eos]}

	def __len__(self):
		return len(self.pairs)

	def truncate_or_pad(self, sent, bos=False, eos=False):
		n_special = (1 if bos else 0) + (1 if eos else 0)
		if len(sent) > self.max_len-n_special: return ([self.lang1.word2index['BOS']] if bos else []) + sent[:self.max_len-n_special] + ([self.lang1.word2index['EOS']] if eos else [])
		else: return ([self.lang1.word2index['BOS']] if bos else []) + sent + ([self.lang1.word2index['EOS']] if eos else []) + [self.lang1.word2index['PAD']] * (self.max_len - n_special - len(sent))

def collate_fn(batch):
	source = torch.tensor([b['indexed'][0] for b in batch])
	target_bos = torch.tensor([b['indexed'][1] for b in batch])
	target_eos = torch.tensor([b['indexed'][2] for b in batch])
	return source, target_bos, target_eos

