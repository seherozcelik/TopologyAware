import torch
import os

import numpy as np
import pydicom
import cv2

from model import UNet   

def prepare_gold(label_name):
    goldImage = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE) 
    
    goldImage = goldImage.astype(np.float32)

    return goldImage

def normalizeImage(img):

    normImg = np.zeros(img.shape) 
    for i in range(img.shape[0]):
        if img[i, :, :].std() != 0:
            normImg[i, :, :] = (img[i, :, :] - img[i, :, :].mean()) / (img[i, :, :].std())

    return normImg.astype(np.float32)

def getData(folder, gold_folder):

    image_names = os.listdir(folder)

    data = []

    for name in image_names:
        im = pydicom.dcmread(folder + '/' + name).pixel_array
        im = np.expand_dims(im,0)
        im = normalizeImage(im)

        label = prepare_gold(gold_folder + '/' + name.split('.')[0] + '.png')

        data.append([im, label])

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
 
