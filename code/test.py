from lang import Lang
from utils import *
from dataset import SentencePairDataset, collate_fn, create_datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
	lang1 = Lang('english', [['hello', 'fda', 'fsdf'], ['okf', 'fda', 'fdsfa']])
	print(lang1)

	# dataset = SentencePairDataset('cmn.txt', 'eng', 'chi')
	# print(len(dataset))
	# print(dataset.lang1)
	# print(dataset.lang2)
	# print(dataset[100])
	# print(dataset)
	train_dataset, test_dataset = create_datasets('../data/cmn.txt', 'english', 'chinese', 5, 0.1)
	print(len(train_dataset), len(test_dataset))

	dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
	for i, (source, target_bos, target_eos) in enumerate(dataloader):
		if i == 0: print(source, target_bos, target_eos)
	print(len(train_dataset.lang1), len(train_dataset.lang2))
	print(len(test_dataset.lang1), len(test_dataset.lang2))