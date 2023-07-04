import torch.nn as nn
import torch
import numpy as np
from ripser import ripser
import helper_functions as hp
import persim
##############################
import cv2
##################################
import gudhi.wasserstein
import traceback

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, golds, typ, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f):
        gold = torch.FloatTensor(golds[0][0,:,:]).cuda()
        pred = pred[0,0,:,:]
     
        loss = - gold * torch.log(pred + 1e-8) - (1-gold) * torch.log((1-pred) + 1e-8)  
                 
        bd0=0
        bd1=0
        
        pairs = hp.getPairs(pred)

        if pairs != None:
            if (typ == 's' and alpha_s > 0) or (typ == 'm' and alpha_m > 0) or (typ == 'f' and alpha_f > 0):
                with open(golds[1][0], 'rb') as f:
                    dgms0 = np.load(f)
                    
                predDgms0 = pairs[0]
                predDgms0 = predDgms0.tolist()
                predDgms0 = np.array(predDgms0)

                bd0 = gudhi.wasserstein.wasserstein_distance(predDgms0, dgms0, order=1., internal_p=2.)
                
            if (typ == 's' and beta_s > 0) or (typ == 'm' and beta_m > 0) or (typ == 'f' and beta_f > 0):
                with open(golds[2][0], 'rb') as f:
                    dgms1 = np.load(f) 
                    
                predDgms1 = pairs[1]
                predDgms1 = predDgms1.tolist()
                predDgms1 = np.array(predDgms1)

                bd1 = gudhi.wasserstein.wasserstein_distance(predDgms1, dgms1, order=1., internal_p=2.)     
                
        if typ == 's':
            weight = (1.0 + alpha_s * bd0 + beta_s * bd1)
        elif typ == 'm':
            weight = (1.0 + alpha_m * bd0 + beta_m * bd1)
        else:
            weight = (1.0 + alpha_f * bd0 + beta_f * bd1)
              
        return weight * torch.mean(loss)