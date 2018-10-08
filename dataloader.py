from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

def get_imdb(batch_size):
	TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
	LABEL = data.Field(sequential=False)


	# make splits for data
	train, test = datasets.IMDB.splits(TEXT, LABEL)

	# print information about the data
	print('train.fields', train.fields)
	print('len(train)', len(train))
	print('vars(train[0])', vars(train[0]))

	# build the vocabulary
	TEXT.build_vocab(train, vectors=GloVe(name='42B', dim=300,max_vectors=500000))
	LABEL.build_vocab(train)

	# print vocab information
	print('len(TEXT.vocab)', len(TEXT.vocab))
	print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

	# make iterator for splits
	train_iter, test_iter = data.BucketIterator.splits(
	    (train, test), batch_size=batch_size,device='cuda:0')

	return train_iter, test_iter,vectors

if __name__ == "__main__":
	train, test = get_imdb(5)
	b = next(train)


