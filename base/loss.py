import torch.nn as nn
import torch
import numpy as np
from ripser import ripser
import train as tr
##############################
import cv2
##################################

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, golds):
        gold = torch.FloatTensor(golds[0,:,:]).cuda()
        pred = pred[0,0,:,:]

        loss = - gold * torch.log(pred + 1e-8) - (1-gold) * torch.log((1-pred) + 1e-8)
    
        return torch.mean(loss)