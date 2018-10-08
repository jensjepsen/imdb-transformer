import torch

def get_pos_onehot(length):
    onehot = torch.zeros(length,length)
    idxs = torch.arange(length).long().view(-1,1)
    onehot.scatter_(1,idxs,1)
    return onehot

if __name__ == "__main__":
    print get_pos_onehot(3)
