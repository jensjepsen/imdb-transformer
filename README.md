# Transformer Networks for Sentiment Analysis

Implements a simple binary classifier for sentiment analysis, embedding sentences using a Transformer network. Transformer networks were introduced in the paper [All You Need is Attention](https://arxiv.org/abs/1706.03762), where the authors achieve state of the art performance on several NLP tasks.

## Usage
Run `python train.py`, to train a model on the IMDB reviews dataset (it will be downloaded automatically through `torchtext` if it's not present). This uses trained positional embeddings for the transformer networks, as opposed to the sinusoidal positional encodings introduced in the paper. 

To use the `Transformer` module in another project, be sure to add some sort of positional encoding to the input before passing it to the module, as these are not automatically added. 

#### Options
```
python train.py --help
usage: train.py [-h] [--max_length MAX_LENGTH] [--model_size MODEL_SIZE]
                [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                [--device DEVICE] [--num_heads NUM_HEADS]
                [--num_blocks NUM_BLOCKS] [--dropout DROPOUT]
                [--train_word_embeddings TRAIN_WORD_EMBEDDINGS]
                [--batch_size BATCH_SIZE]

Train a Transformer network for sentiment analysis

optional arguments:
  -h, --help            show this help message and exit
  --max_length MAX_LENGTH
                        Maximum sequence length, sequences longer than this
                        are truncated
  --model_size MODEL_SIZE
                        Hidden size for all hidden layers of the model
  --epochs EPOCHS       Number of epochs to train for
  --learning_rate LEARNING_RATE
                        Learning rate for optimizer
  --device DEVICE       Device to use for training and evaluation e.g. (cpu,
                        cuda:0)
  --num_heads NUM_HEADS
                        Number of attention heads in the Transformer network
  --num_blocks NUM_BLOCKS
                        Number of blocks in the Transformer network
  --dropout DROPOUT     Dropout (not keep_prob, but probability of ZEROING
                        during training, i.e. keep_prob = 1 - dropout)
  --train_word_embeddings TRAIN_WORD_EMBEDDINGS
                        Train GloVE word embeddings
  --batch_size BATCH_SIZE
                        Batch size
```

## Requirements
- Python 2.7
- PyTorch 4.1
- TorchText
- NumPy
- tqdm (optional)

## Acknowledgements
The Transformer networks were introduced by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin in [All You Need is Attention](https://arxiv.org/abs/1706.03762).