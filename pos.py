import torch

def get_pos_onehot(length):
	onehot = torch.zeros(length).view(length,length)
	idxs = torch.arange(length).view(-1,1)
	onehot.scatter_(1,idxs,1)
	return onehot
