import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from model import Net

DEVICE = "cuda:0"

import visdom

def num2words(vocab,vec):
    return [vocab.itos[i] for i in vec]


vis = visdom.Visdom()

def plot_weights(model,windows,b):
    try:
        weights = model.transformer.blocks[0].attention.weights.to("cpu").numpy()
    except AttributeError:
        print "No weights yet"
        return None
    text,dims = b
    if windows is None:
        windows = [None] * weights.shape[0]
    new_windows = []
    weights = weights[0]
    names = num2words(vocab,test[0].numpy)
    dims = dims[0].item()
    weights = weights[:,:dims,:dims]

    for weight, window in zip(weights,windows):
        print
        new_windows.append(vis.heatmap(weight,column_names=names,row_names=names,win=window))
    return new_windows


def val(model,test,vocab):
    model.eval()
    visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i,b in enumerate(test):
            if i == 0:
                visdom_windows = plot_weights(model,visdom_windows,b)

            model_out = model(b.text[0].to(DEVICE)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        print "{}%, {}/{}".format(correct / total,correct,total)

def train():
    train, test, vectors, vocab = get_imdb(128,max_length=500)
    epochs = 1000

    model = Net(embeddings=vectors,max_length=500).to(DEVICE)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_mavg = 0.0
    for i in xrange(epochs):
        val(model,test)
        model.train()
        for j,b in enumerate(iter(train)):
            optimizer.zero_grad()
            model_out = model(b.text[0].to(DEVICE))
            loss = criterion(model_out,b.label.to(DEVICE))
            loss.backward()
            optimizer.step()
            loss_mavg = loss_mavg * 0.9 + loss.item()
        print "Epoch {}, Batch {}, Loss {}".format(i,j,loss_mavg)

if __name__ == "__main__":
    train()



