class Lang: 
	def __init__(self, language, sentences=None):
		self.language = language
		self.word2index = {}
		self.index2word = {}
		self.wordCount = {}
		self.vocabCount = 0
		self.addWord('PAD')
		self.addWord('BOS')
		self.addWord('EOS')
		self.addWord('UNK')
		if sentences != None: self.createVocab(sentences)

	def __str__(self):
		return  '---------------------------\n' + \
				'language: %s\n' % self.language + \
				'word2index: %s\n' % list(self.word2index.items())[:10] + \
				'index2word: %s\n' % list(self.index2word.items())[:10]+ \
				'wordCount: %s\n' % list(self.wordCount.items())[:10] + \
				'vocabCount: %s\n' % self.vocabCount + \
				'---------------------------\n'

	def __len__(self):
		return len(self.word2index)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.vocabCount
			self.index2word[self.vocabCount] = word
			self.wordCount[word] = 1
			self.vocabCount += 1
		else: 
			self.wordCount[word] += 1

	def createVocab(self, sentences):
		for sentence in sentences:
			for word in sentence:
				self.addWord(word)


