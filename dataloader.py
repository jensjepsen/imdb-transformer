from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import string

def tokenize(input):
    """
        Naive tokenizer, that lower-cases the input
        and splits on punctuation and whitespace
    """
    input = input.lower()
    for p in string.punctuation:
        input = input.replace(p," ")
    return input.strip().split()


def num2words(vocab,vec):
    """
        Converts a vector of word indicies
        to a list of strings
    """
    return [vocab.itos[i] for i in vec]

def get_imdb(batch_size,max_length):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,tokenize=tokenize,fix_length=max_length)
    LABEL = data.Field(sequential=False,unk_token=None,pad_token=None)


    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))
    print('vars(test[0])',vars(test[0]))

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='42B', dim=300,max_vectors=500000))
    LABEL.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size)

    return train_iter, test_iter, TEXT.vocab.vectors, TEXT.vocab

if __name__ == "__main__":
        """
        If run seperately, does a simple sanity check,
        by printing different values,
        and counting labels
    """
    train, test, vectors,vocab = get_imdb(1,50)
    from collections import Counter
    print list(enumerate(vocab.itos[:100]))
    cnt = Counter()
    for b in iter(train):
        print num2word(b.text[0].numpy())
        cnt[b.label[0].item()] += 1
    print cnt

