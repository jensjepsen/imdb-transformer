import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from model import Net

try:
    # try to import visdom for visualisation of attention weights
    import visdom
    from helpers import plot_weights
    vis = visdom.Visdom()
except ImportError:
    vis = None
    pass

def val(model,test,vocab,device):
    """
        Evaluates model on the test set
    """
    model.eval()
    if not vis is None:
        visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i,b in enumerate(test):
            if not vis is None and i == 0:
                visdom_windows = plot_weights(model,visdom_windows,b,vocab,vis)

            model_out = model(b.text[0].to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        print "{}%, {}/{}".format(correct / total,correct,total)

def train(max_length,model_size,
            epochs,learning_rate,
            device,num_heads,num_blocks,
            dropout,train_word_embeddings,
            batch_size):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    train, test, vectors, vocab = get_imdb(batch_size,max_length=max_length)

    model = Net(
                model_size=model_size,embeddings=vectors,
                max_length=max_length,num_heads=num_heads,
                num_blocks=num_blocks, dropout=dropout,
                train_word_embeddings=train_word_embeddings,
                ).to(device)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    loss_mavg = 0.0
    for i in xrange(0,epochs+1):
        model.train()
        for j,b in enumerate(iter(train)):
            optimizer.zero_grad()
            model_out = model(b.text[0].to(device))
            loss = criterion(model_out,b.label.to(device))
            loss.backward()
            optimizer.step()
            loss_mavg = loss_mavg * 0.9 + loss.item()
        print "Epoch {}, Batch {}, Loss {}".format(i,j,loss_mavg)

        # Validate on test-set every epoch
        val(model,test,vocab,device)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")
    
    ap.add_argument("--max_length",default=500,type=int,help="Maximum sequence length, \
                                                                    sequences longer than this are truncated")
    
    ap.add_argument("--model_size",default=128,type=int,help="Hidden size for all \
                                                                    hidden layers of the model")
    
    ap.add_argument("--epochs",default=1000,type=int,help="Number of epochs to train for")

    ap.add_argument("--learning_rate",default=0.001,type=float,dest="learning_rate",help="Learning rate for optimizer")
    
    ap.add_argument("--device",default="cuda:0",dest="device",help="Device to use for training \
                                                                    and evaluation e.g. (cpu, cuda:0)")
    ap.add_argument("--num_heads",default=4,type=int,dest="num_heads",help="Number of attention heads in \
                                                                    the Transformer network")
    
    ap.add_argument("--num_blocks",default=1,type=int,dest="num_blocks",help="Number of blocks in the Transformer network")
    
    ap.add_argument("--dropout",default=0.1,type=float,dest="dropout",help="Dropout (not keep_prob, but probability of ZEROING \
                                                                    during training, i.e. keep_prob = 1 - dropout)")

    ap.add_argument("--train_word_embeddings",type=bool,default=True,dest="train_word_embeddings",help="Train GloVE word embeddings")
    
    ap.add_argument("--batch_size",type=int,default=128,help="Batch size")
    args = vars(ap.parse_args())

    train(**args)



