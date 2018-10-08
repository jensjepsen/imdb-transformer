import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from model import Net

DEVICE = "cuda:0"

import visdom

def plot_weights(model,windows):
    weights = model.transformer.blocks[0].attention.weights.to("cpu").numpy()
    weights = weights[0]
    for weight, window in zip(weights,windows):
        visdom.heatmap(weights)


def val(model,test):
    model.eval()
    visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i,b in enumerate(test):
            if i == 0:
                plot_weights(model,visdom_windows)
            model_out = model(b.text[0].to(DEVICE)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        print "{}%, {}/{}".format(correct / total,correct,total)

def train():
    train, test, vectors = get_imdb(128,max_length=500)
    epochs = 1000

    model = Net(embeddings=vectors,max_length=500).to(DEVICE)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad))
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



