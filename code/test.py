from lang import Lang
from utils import *
from dataset import SentencePairDataset, collate_fn
from torch.utils.data import DataLoader

if __name__ == '__main__':
	lang1 = Lang('english', [['hello', 'fda', 'fsdf'], ['okf', 'fda', 'fdsfa']])
	print(lang1)

	dataset = SentencePairDataset('cmn.txt', 'eng', 'chi')
	print(len(dataset))
	print(dataset.lang1)
	print(dataset.lang2)
	print(dataset[100])
	print(dataset)

	dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
	for i, (source, target_bos, target_eos) in enumerate(dataloader):
		if i == 0: print(source, target_bos, target_eos)