from utils import read_config
from model import Encoder, Decoder
from dataset import SentencePairDataset, collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import random

from tensorboardX import SummaryWriter

global device

def train(config):
	train_config = config['train']
	
	device = train_config['device']
	tqdm.write('Training on {}'.format(device))
	writer = SummaryWriter('log')

	train_dataset = SentencePairDataset(**config['dataset'], mode='train')
	test_dataset  = SentencePairDataset(**config['dataset'], mode='test')

	train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
	test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)

	encoder = Encoder(vocab_size=len(train_dataset.lang1), **config['encoder'], device=device).to(device)
	decoder = Decoder(vocab_size=len(train_dataset.lang2), **config['decoder']).to(device)

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=train_config['lr'])
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=train_config['lr'])

	criterion = nn.NLLLoss()


	tqdm.write('[-] Start training! ')
	epoch_bar = tqdm(range(train_config['n_epochs']), desc='[Total progress]', leave=True, position=0, dynamic_ncols=True)
	for epoch in epoch_bar:
		batch_bar = tqdm(range(len(train_dataloader)), desc='[Train epoch {:2}]'.format(epoch), leave=True, position=0, dynamic_ncols=True)
		for batch in batch_bar:
			(source, target_bos, target_eos) = next(iter(train_dataloader))
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			source, target_bos, target_eos = source.to(device), target_bos.to(device), target_eos.to(device)
			encoder_output, encoder_hidden = encoder(source)
			decoder_output = decoder(target_bos, encoder_hidden)

			loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), target_eos.view(-1))
			n_hit, n_total = hitRate(decoder_output, target_eos)
			loss.backward()
			#print(loss.item())

			encoder_optimizer.step()
			decoder_optimizer.step()
			
			batch_bar.set_description('[Train epoch {:2} | Loss: {:.2f} | Hit: {}/{}]'.format(epoch, loss, n_hit, n_total))
		writer.add_scalar('train_loss', loss.item(), epoch)

		batch_bar = tqdm(range(len(test_dataloader)), desc='[Test epoch {:2}]'.format(epoch), leave=True, position=0, dynamic_ncols=True)
		for batch in batch_bar:
			(source, target_bos, target_eos) = next(iter(test_dataloader))
			source, target_bos, target_eos = source.to(device), target_bos.to(device), target_eos.to(device)
			
			with torch.no_grad():
				encoder_output, encoder_hidden = encoder(source)
				decoder_output = decoder(target_bos, encoder_hidden)
				loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), target_eos.view(-1))
				n_hit, n_total = hitRate(decoder_output, target_eos)
				batch_bar.set_description('[Test epoch {:2} | Loss: {:.2f} | Hit: {}/{}]'.format(epoch, loss, n_hit, n_total))

		writer.add_scalar('test_loss', loss.item(), epoch)
		sample(test_dataset, encoder, decoder)

	tqdm.write('[-] Done!')

def hitRate(decoder_output, target_eos):

	decoder_output = decoder_output.view(-1, decoder_output.size(-1))
	decoder_output = decoder_output.topk(1)[1].view(-1)
	target_eos = target_eos.view(-1)
	n_hit = (decoder_output == target_eos).sum()
	#print(n_hit)
	n_total = target_eos.size(0)
	return n_hit, n_total

def sample(dataset, encoder, decoder):
	rand_idx = random.randint(0, len(dataset)-1)
	data = dataset[rand_idx]
	source, target_bos, target_eos = data['indexed']
	source, target_bos, target_eos = torch.tensor(source).unsqueeze(0).to(device), torch.tensor(target_bos).unsqueeze(0).to(device), torch.tensor(target_eos).unsqueeze(0).to(device)
	encoder_output, encoder_hidden = encoder(source)
	decoder_output = decoder(target_bos, encoder_hidden)
	decoder_output = decoder_output.topk(1)[1].view(-1)
	sentence = ''.join([dataset.lang2.index2word[i.item()] for i in decoder_output])

	print('> ', ' '.join(data['org'][0]))
	print('= ', ''.join(data['org'][1]))
	print('< ', sentence)






if __name__ == '__main__':
	config = read_config() # Read config from 'config.yaml' to dictionary
	train(config)
