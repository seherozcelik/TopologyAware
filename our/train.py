import torch
import torch.optim as optim
import torch.utils.data as data
import time

from copy import deepcopy

import numpy as np
import cv2

import helper_functions as hp
from model import UNet
from loss import WeightedCrossEntropyLoss

import json

import sys

def model_eval(model, data_loader, loss_func, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f):
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)
    loss = 0
    model.eval()
    with torch.no_grad():
        for input, label in data_loader:    
            pred = model(input[0].cuda())
            typ = fileTypesDict[input[1][0]]
            loss += loss_func(pred, label, typ, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f)
        avrg_loss = loss / len(data_loader)  
        return avrg_loss   

def train(in_channel, first_out_channel, trn_folder, val_folder, gold_folder, lr, patience, min_delta, model_name, data_type,
         alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f, initial_model):
    
    model = UNet(in_channel,first_out_channel).cuda()
    model.load_state_dict(torch.load(initial_model))

    loss_func = WeightedCrossEntropyLoss()

    optimizer = optim.Adadelta(model.parameters(),lr)
    optimizer.load_state_dict(torch.load(initial_model.split('.pth')[0]+'_optim.pth'))
   
    train_loader = data.DataLoader(hp.getData(trn_folder, gold_folder, data_type),batch_size=1, shuffle=True)
    val_loader = data.DataLoader(hp.getData(val_folder, gold_folder, data_type), batch_size=1)
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)

    init_tr_loss = model_eval(model, train_loader, loss_func, 0.000005, 0, 0.000005, 0, 0, 0.0001);
    print("initial tr loss: ",init_tr_loss)
    init_val_loss = model_eval(model, val_loader, loss_func, 0.000005, 0, 0.000005, 0, 0, 0.0001);
    print("initial val loss: ",init_val_loss)
    
    
    losses = []
    val_losses = []
    min_val_loss = np.Inf
    cnter = 0
    l = len(train_loader)
    tot_time_passed = 0
    for epoch in range(10000):
        start_time = time.time()
        loss_sum = 0
        model.train()
        for input, label in train_loader:
            output = model(input[0].cuda())
      
            optimizer.zero_grad()
        
            typ = fileTypesDict[input[1][0]]
            loss = loss_func(output,label, typ, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f)

            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        trainingLoss = loss_sum / l    
        losses.append(trainingLoss)
        val_loss = model_eval(model, val_loader, loss_func, alpha_s, beta_s, alpha_m, beta_m, alpha_f, beta_f);
        val_losses.append(val_loss.item())
        
        if min_val_loss > val_loss.item() + min_delta:
            min_val_loss = val_loss.item()
            cnter = 0
            weights = deepcopy(model.state_dict())
            used_loss = min_val_loss
        else:
            cnter += 1  

        time_passed = time.time() - start_time 
        tot_time_passed = tot_time_passed + time_passed
        print('epoch: ', epoch, '- trn_loss: ', round(trainingLoss,10), '- val_loss: ', round(val_loss.item(),10), '- counter: ', cnter, '- second_passed: ', round(time_passed), '-tot mins: ', round(tot_time_passed/60))
        
        if cnter >=patience:
            torch.save(weights, model_name)
            print('used loss: ', used_loss)
            break 

    return losses, val_losses  