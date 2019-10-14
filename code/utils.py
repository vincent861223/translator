from nltk import word_tokenize
from tqdm import tqdm
import yaml

def read_datafile(filename, mode='train'):
	with open(filename) as f:
		lines = f.read().split('\n')
		if mode =='train': lines = lines[:-len(lines)//10]
		elif mode == 'test': lines = lines[-len(lines)//10:]
		pairs = []
		tqdm.write('Loading {} corpus from {} ...'.format(mode + 'ing', filename))
		lines = tqdm(lines, desc='Tokenizing sentences', leave=False, dynamic_ncols=True)
		for line in lines:
			pair = line.split('\t')
			if len(pair) == 2: 
				pairs.append([word_tokenize(pair[0]), tokenize(pair[1])])
		tqdm.write('Loaded {} sentences'.format(len(pairs)))
	return pairs 

def read_config():
	with open('config.yaml') as f:
		config = yaml.load(f)
	return config

def tokenize(sentence):
	return [w for w in sentence]