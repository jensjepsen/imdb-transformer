from torch import optim
from torch import nn
from dataloader import get_imdb
from model import Net

DEVICE = "cuda:0"

def train():
	train, test, vectors = get_imdb(128)
	epochs = 1000

	model = Net(embeddings=vectors).to(DEVICE)
	optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad))
	criterion = nn.CrossEntropyLoss()

	loss_mavg = 0.0
	for i in xrange(epochs):
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



