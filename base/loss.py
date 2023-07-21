import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, gold):
        gold = torch.FloatTensor(gold[0,:,:]).cuda()
        pred = pred[0,0,:,:]

        loss = - gold * torch.log(pred + 1e-8) - (1-gold) * torch.log((1-pred) + 1e-8)
    
        return torch.mean(loss)
