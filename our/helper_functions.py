import torch
import os

import numpy as np
import pydicom
import cv2

import skimage.segmentation as sg

from model import UNet       

import ripserplusplus as rpp

import json

def getXYofboundary(im):

    bnd = sg.find_boundaries(im, mode='inner').astype(np.uint8)
    
    return getXY(bnd)

def getXY(bnd):
    return np.transpose(np.nonzero(bnd))

def getPairs(output, maxdim = 1):
    pairs = None
    pred = output.detach().cpu().numpy()
    pred = np.uint8(np.where(pred >= 0.5,1,0))
    xy = getXYofboundary(pred)
    if xy.shape[0]>0:
        pairs= rpp.run("--format point-cloud --dim "+str(maxdim), xy)
        
    return pairs

def prepare_gold_and_weights(label_name, dgms0_name, dgms1_name):
    goldImage = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE) 
    
    goldImage = goldImage.astype(np.float32)

    return [goldImage, dgms0_name, dgms1_name]

def normalizeImage(img):

    normImg = np.zeros(img.shape) 
    for i in range(img.shape[0]):
        if img[i, :, :].std() != 0:
            normImg[i, :, :] = (img[i, :, :] - img[i, :, :].mean()) / (img[i, :, :].std())

    return normImg.astype(np.float32)

def getData(folder, gold_folder, data_type):
    
    with open('../../data/vesseltypes.json') as f:
        fileTypesDict = json.load(f)

    image_names = os.listdir(folder)

    data = []

    for name in image_names:
        if data_type == 'all' or (data_type == 'big' and (fileTypesDict[name] == 'f' or fileTypesDict[name] == 'm')
                                 ) or (data_type == 'small' and (fileTypesDict[name] == 's' or fileTypesDict[name] == 'm')):
            im = pydicom.dcmread(folder + '/' + name).pixel_array
            im = np.expand_dims(im,0)
            im = normalizeImage(im)
            inpt = []
            inpt.append(im)
            inpt.append(name)

            label = prepare_gold_and_weights(gold_folder + '/' + name.split('.')[0] + '.png',
                                             gold_folder + '/dgms0/' + name.split('.')[0] + '.npy',
                                             gold_folder + '/dgms1/' + name.split('.')[0] + '.npy')

            data.append([inpt, label])

    return data   

def test(in_channel, first_out_channel, model_name, tst_im_name, gold_im_name):
       
    model = UNet(in_channel,first_out_channel).cuda()

    model.load_state_dict(torch.load(model_name))
        
    model.eval()
    
    with torch.no_grad():  
        im_tst = pydicom.dcmread(tst_im_name).pixel_array
        im = np.expand_dims(im_tst,0)
        im = normalizeImage(im)
        im = np.expand_dims(im,0)

        label = cv2.imread(gold_im_name,cv2.IMREAD_GRAYSCALE)

        output = model(torch.FloatTensor(im).cuda())

        pre_prediction = output.detach().cpu().numpy()[0,0,:,:]
        prediction = np.uint8(np.where(pre_prediction >= 0.5,1,0))

        return im_tst, pre_prediction, prediction, label
 