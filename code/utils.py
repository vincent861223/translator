from nltk import word_tokenize
from tqdm import tqdm
import yaml

def read_datafile(filename, mode='train'):
	"""
	Usage: Read datafile from 'filename' and tokenize each sentence into a list of words 
	Return: 
		pairs: a pair of a sentence and its corresponding translation 
				Ex. pairs = [
								[['How', 'are', 'you', '?'], ['我', '很', '好']],
								[['Hi', '.'], ['嗨', '。']]
								.
								.
								.
								[['Goodbye', '!'], ['再', '見']]
							]
	"""
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
	"""
	Read config file and return a dict.
	Return: dict
	"""
	with open('config.yaml') as f:
		config = yaml.load(f)
	return config

def tokenize(sentence):
	return [w for w in sentence]



