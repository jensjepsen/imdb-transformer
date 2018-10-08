from torch import optim
from dataloader import get_imdb
DEVICE = "cuda:0"

def train():
	train, test, vectors = get_imdb()
	epochs = 1000

	model = Net(embeddings=vectors).to(DEVICE)
	optimizer = optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()

	loss_mavg = 0.0
	for i in xrange(epochs):
		for j,b in enumerate(iter(train)):
			loss.zero_grads()
			model_out = model(b.text)
			loss = criterion(model_out,b.labels)
			loss.backward()
			optimizer.step()
			loss_mavg = loss_mavg * 0.9 + loss.item()
			if j % 100:
				print "Epoch {}, Batch {}, Loss {}".format(i,j,loss_mavg)



if __name__ == "__main__":
	train()



