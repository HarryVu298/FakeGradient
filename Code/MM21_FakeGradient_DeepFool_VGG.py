'''
Modified in 4/4/2021 1:18
Collect the efficiency result based on Google Net and DeepFool.
Modification:
    Adding scaling factor when modifying the CNN FC layer.
    Checking the DeepFool performance

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
from ModelModify import ModifyModel,ModifyModelVGG, ModifyModelScale,ModifyModelVGGScale
from DeepFoolC import deepfoolC
from DeepFoolB import deepfoolB
import HeatMapForgradientOrPerturbation as HM
#from HeatMapForgradientOrPerturbation import heatmap
import cv2
from scipy.misc import imread, imsave, imresize

Scale=20


net = models.vgg19(pretrained=True).cuda()
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net.eval()
'''
net2 = models.resnet34(pretrained=True)
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net2.eval()
'''

net2 = models.vgg19(pretrained=True)
net2= ModifyModelVGGScale(net2,Scale)
net2.cuda()
net2.eval()


#
AT="DeepFool"
CSVfilenameTime ='VGG19'+'_'+ AT +"_"+str(Scale)+"_MethodB"+'_Result.csv'
fileobjT = open(CSVfilenameTime, 'w', newline='')  # 
# fileobj.write('\xEF\xBB\xBF')#
# 
writerT = csv.writer(fileobjT)  # csv.writer(fileobj)writer writer
ValueTime=['Original ATT,GT','Original ATT, ATT','On Fake ATT, GT','On Fake ATT,ATT','On Fake ATT, Def','ACC','ACC_ALL','DL2R','DL2G','DL2B','DLIR','DLIG','DLIB','AL2R','AL2G','AL2B','ALIR','ALIG','ALIB']
writerT.writerow(ValueTime)
CountT=0        #deepfool
CountTotal=0    #
CountDF_EFF=0   #deepfool 
CountDF_EFF_Def=0  #DeepFool


Folder='D:/workspace/imagenet2012B/test/'
FileName='ILSVRC2012_test'
Append='.JPEG'            #00099990
Error=[]
for i in range(1,100000):
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
    #r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
    '''
    f_image = net.forward(Variable(im[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    Originallabel = I[0]
    '''
    r, loop_i, label_orig, label_pert, Originallabel,Protected,pert_image,TheGradient = deepfoolC(im, net2)
    rB, loop_iB, label_origB, label_pertB, pert_imageB,TheGradientB = deepfoolB(imB, net)
    print("original:    ", Originallabel)
    print("original:    ", Protected)
    #summary result
    print("Original Attack Result:  ", label_pertB, "    Original Label in original Attack: ",label_origB)
    print(" Attack Result In Fake:  ", label_pert, "   Original Label in Attack With Fake: ", Originallabel,"   Protected By Fake: ",Protected )
    Acc=0
    AccB=0
    if label_pertB!=label_origB:
        print("DeepFool Works!")
        CountDF_EFF=CountDF_EFF+1
        if label_origB==Protected:
            CountDF_EFF_Def=CountDF_EFF_Def+1
            Acc=1
    if label_origB == Protected:
        CountT=CountT+1
        AccB=1
    print("Efficiency: ===>", int(CountT*100/CountTotal))
    #L2 and Linfinity

    # get the perturbation and the gradient based on the defence one
    RA, RB, RC = HM.get2Dfrom3D(224, 224, r)  # perturbation
    # get the perturbation and the gradient based on the original version
    BRA, BRB, BRC = HM.get2Dfrom3D(224, 224, rB)  # perturbation

    #defence
    L2RD=HM.L2NormValue(RA)
    L2GD = HM.L2NormValue(RB)
    L2BD = HM.L2NormValue(RC)

    LIRD = HM.L_Inf(RA)
    LIGD = HM.L_Inf(RB)
    LIBD = HM.L_Inf(RC)

    #original

    L2RA = HM.L2NormValue(BRA)
    L2GA = HM.L2NormValue(BRB)
    L2BA = HM.L2NormValue(BRC)

    LIRA = HM.L_Inf(BRA)
    LIGA = HM.L_Inf(BRB)
    LIBA = HM.L_Inf(BRC)
    '''
    ValueTime = ['Original ATT,GT', 'Original ATT, ATT', 'On Fake ATT, GT', 'On Fake ATT,ATT', 'On Fake ATT, Def',
                 'ACC', 'ACC_ALL', 'DL2R', 'DL2G', 'DL2B', 'DLIR', 'DLIG', 'DLIB', 'AL2R', 'AL2G', 'AL2B', 'ALIR',
                 'ALIG', 'ALIB']'''
    ValueTime = [label_origB, label_pertB, Originallabel,label_pert, Protected,
                 Acc, AccB, L2RD, L2GD, L2BD, LIRD, LIGD, LIBD,
                 L2RA, L2GA, L2BA, LIRA, LIGA, LIBA]
    writerT.writerow(ValueTime)



print("Final Result:   ",CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def)
ValueTime=[CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def,int(CountT*100/CountTotal)]
writerT.writerow(ValueTime)
exit()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)
'''
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                        transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])
'''
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

plt.figure()

plt.xticks([])
plt.yticks([])

plt.imshow(tf(pert_image.cpu()[0]))
str_label_pert="Perturbed based on the fake CNN"
plt.title(str_label_pert)
plt.show()


# get the perturbation and the gradient based on the defence one
GA,GB,GC=HM.get2Dfrom3D(224,224,TheGradient)   #gradient
RA,RB,RC=HM.get2Dfrom3D(224,224,r)      # perturbation
# get the perturbation and the gradient based on the original version
BGA,BGB,BGC=HM.get2Dfrom3D(224,224,TheGradientB)   #gradient
BRA,BRB,BRC=HM.get2Dfrom3D(224,224,rB)      # perturbation

#
title="Perturbation Compare A, Positive 4"
HM.CVShowCompareFB(RA,BRA,title)
title="Perturbation Compare A, Negtive 4"
HM.CVShowCompareGB(RA,BRA,title)
print("L2")
print(HM.L2NormValue(RA))
print(HM.L2NormValue(BRA))
title="Perturbation Compare A, Positive 0"
HM.CVShowCompareF(RA,BRA,title)
title="Perturbation Compare A, Negtive 0"
HM.CVShowCompareG(RA,BRA,title)

title="Perturbation Compare A, Positive 2"
HM.CVShowCompareFC(RA,BRA,title)
title="Perturbation Compare A, Negtive 2"
HM.CVShowCompareGC(RA,BRA,title)


title="Perturbation Compare B, Positive"
HM.CVShowCompareF(RB,BRB,title)
title="Perturbation Compare B, Negtive"
HM.CVShowCompareG(RB,BRB,title)

title="Perturbation Compare C, Positive"
HM.CVShowCompareF(RC,BRC,title)
title="Perturbation Compare C, Negtive"
HM.CVShowCompareG(RC,BRC,title)

title="Gradient Compare A, Positive"
HM.CVShowCompareF(GA,BGA,title)
title="Gradient Compare A, Negtive"
HM.CVShowCompareG(GA,BGA,title)


title="Gradient Compare B, Positive"
HM.CVShowCompareF(GB,BGB,title)
title="Gradient Compare B, Negtive"
HM.CVShowCompareG(GB,BGB,title)


title="Gradient Compare C, Positive"
HM.CVShowCompareF(GC,BGC,title)
title="Gradient Compare C, Negtive"
HM.CVShowCompareG(GC,BGC,title)



#exit()

#show the heatmap of the perturbation


RA,RB,RC=HM.get2Dfrom3D(224,224,r)
Title="Perturbation Orginal A"
im, cbar = HM.heatmap(RA,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation B"
im, cbar = HM.heatmap(RB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation C"
im, cbar = HM.heatmap(RC,Title,cmap="YlGn", cbarlabel="Perturbation")


RA,RB,RC=HM.get2Dfrom3D(224,224,rB)
Title="Perturbation by Faked GradientA"
im, cbar = HM.heatmap(RA,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation B"
im, cbar = HM.heatmap(RB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation C"
im, cbar = HM.heatmap(RC,Title,cmap="YlGn", cbarlabel="Perturbation")

GA,GB,GC=HM.get2Dfrom3D(224,224,TheGradient)

Title="Gradient A"
im, cbar = HM.heatmap(GA,Title,cmap="YlGn", cbarlabel="Gradient")
Title="Gradient B"
im, cbar = HM.heatmap(GB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Gradient C"
im, cbar = HM.heatmap(GC,Title,cmap="YlGn", cbarlabel="Gradient")
