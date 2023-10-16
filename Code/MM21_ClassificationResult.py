'''
Modified in 4/4/2021 1:18
Collect the efficiency result based on Google Net and DeepFool.
Modification:
    Adding scaling factor when modifying the CNN FC layer.
    Checking the DeepFool performance
4_11
collect result by the two nets
'''


import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
import csv
from ModelModify import ModifyModel,ModifyModelVGG, ModifyModelScale,ModifyModelVGGScale,ModifyModelMobNetV2Scale,ModifyModelDensNetScale
from DeepFoolC import deepfoolC
from DeepFoolB import deepfoolB
import HeatMapForgradientOrPerturbation as HM
#from HeatMapForgradientOrPerturbation import heatmap
import cv2
from scipy.misc import imread, imsave, imresize


import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

import time

Scale=1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#net = models.resnet34(pretrained=True).cuda()
net = models.resnet18(pretrained=True).cuda()
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net.eval()



'''
net2 = models.resnet34(pretrained=True)
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net2.eval()
'''
#net2 = models.resnet34(pretrained=True)
net2 = models.resnet18(pretrained=True)
net2= ModifyModelScale(net2,Scale)
net2.cuda()
net2.eval()

print(count_parameters(net))
print(count_parameters(net2))


#
AT="DeepFool"
CSVfilenameTime ='02ResNet18'+'_'+ AT +"_"+str(Scale)+"_classificationresult"+'_Result.csv'
fileobjT = open(CSVfilenameTime, 'w', newline='')  # 
#
# 
writerT = csv.writer(fileobjT)  # csv.writer(fileobj)
ValueTime=['Original ATT,GT','Original ATT, ATT','On Fake ATT, GT','On Fake ATT,ATT','On Fake ATT, Def','ACC','ACC_ALL','DL2R','DL2G','DL2B','DLIR','DLIG','DLIB','AL2R','AL2G','AL2B','ALIR','ALIG','ALIB']
writerT.writerow(ValueTime)
CountT=0        #
CountTotal=0    #
CountDF_EFF=0   #
CountDF_EFF_Def=0  #




Folder='D:/workspace/imagenet2012B/test/'
FileName='ILSVRC2012_test'
Append='.JPEG'            #00099990
Error=[]
for i in range(1,10000):
    Index=str(i+1)
    K=len(Index)
    IndexFull='_'
    for j in range(8-K):
        IndexFull=IndexFull+str(0)
    IndexFull=IndexFull+Index
    FNAME=Folder+FileName+IndexFull+Append
    #im_orig = Image.open('test_im2.jpg')

    CC = cv2.imread(FNAME)
    #print(im_orig.size)
    a, b, c = CC.shape
    #print(CC.shape, c)

    image = imread(FNAME)
    if (len(image.shape) < 3):
        #print('gray')
        continue
    if c!=3:
        continue

    CountTotal=CountTotal+1

    #im_orig = Image.open('test_im2.jpg')
    #im_orig = Image.open('ILSVRC2012_test_00000002.JPEG')
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    #im_origB = Image.open('ILSVRC2012_test_00000002.JPEG')
    im_orig = Image.open(FNAME)
    im_origB = Image.open(FNAME)

    # Remove the mean
    im = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)
    imB = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_origB)

    im = im.cuda()
    imB = imB.cuda()


    #defence

    start = time.time()
    f_image = net2.forward(Variable(imB[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    end = time.time()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    label = I[0]
    B = (np.array(f_image)[0:1000]).flatten().argsort()[::-1]
    Originallabel = B[0]

    #original
    startB = time.time()
    #f_image2 = net.forward(im[None, :, :, :])
    f_image2 = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    endB = time.time()
    IO = (np.array(f_image2)).flatten().argsort()[::-1]
    labelB = IO[0]

    Acc=0
    ACCfake=0
    if Originallabel==labelB:
        Acc=1
    if label!=labelB:
        ACCfake=1

    ValueTime = [IndexFull,label,Originallabel,labelB,Acc,ACCfake,end-start,endB-startB]
    writerT.writerow(ValueTime)
    print(ValueTime)
