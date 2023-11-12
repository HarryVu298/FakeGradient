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
from ModelModify import ModifyModel,ModifyModelVGG, ModifyModelScale,ModifyModelVGGScale,ModifyModelMobNetV2Scale,ModifyModelDensNetScale
from DeepFoolC import deepfoolC
from DeepFoolB import deepfoolB

import cv2
# from scipy.misc import imread, imsave, imresize
import imageio.v2 as imageio
from torchvision import transforms

Scale=20



net = models.densenet121(pretrained=True).cuda()
#net.classifier=nn.Linear(in_features=1024, out_features=2000, bias=True)
#print(net.classifier)
#exit()
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net.eval()
'''
net2 = models.resnet34(pretrained=True)
#net2 = models.resnet18(pretrained=True)
# Switch to evaluation mode
net2.eval()
'''
net2 = models.densenet121(pretrained=True)
net2= ModifyModelDensNetScale(net2,Scale)
net2.cuda()
net2.eval()

#
AT="DeepFool"
CSVfilenameTime ='Densenet121'+'_'+ AT +"_"+str(Scale)+"_MethodB"+'_Result.csv'
fileobjT = open(CSVfilenameTime, 'w', newline='')  # wb
# fileobj.write('\xEF\xBB\xBF')#
# 
writerT = csv.writer(fileobjT)  # csv.writer(fileobj)writerwriter
ValueTime=['Original ATT,GT','Original ATT, ATT','On Fake ATT, GT','On Fake ATT,ATT','On Fake ATT, Def','ACC','ACC_ALL','DL2R','DL2G','DL2B','DLIR','DLIG','DLIB','AL2R','AL2G','AL2B','ALIR','ALIG','ALIB']
writerT.writerow(ValueTime)
CountT=0        #
CountTotal=0    #
CountDF_EFF=0   #deepfool 
CountDF_EFF_Def=0  #DeepFool


Folder='/users/PMIU0211/minhkhoa29082003/Downloads/test/'
FileName='ILSVRC2012_test'
Append='.JPEG'            #00099990
Error=[]

def L2NormValue(tensor):
    return torch.norm(tensor, p=2).item()

# Define the L infinity norm computation
def L_Inf(tensor):
    return torch.norm(tensor, p=float('inf')).item()

# Define a method to convert a 3D tensor to 2D (assuming the first dimension is channel)
def get2Dfrom3D(height, width, tensor):
    return tensor[0].cpu().numpy(), tensor[1].cpu().numpy(), tensor[2].cpu().numpy()

# Define a method to show heatmap using matplotlib
def heatmap(data, title, cmap='YlGn', cbarlabel=''):
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label=cbarlabel)
    plt.title(title)
    plt.show()

# Define a function to visualize perturbation or gradient comparison
def CVShowCompare(tensor1, tensor2, title):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(tensor1)
    axes[0].set_title('Tensor 1')
    axes[1].imshow(tensor2)
    axes[1].set_title('Tensor 2')
    plt.suptitle(title)
    plt.show()


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
    if CC is None:
        print(f"Failed to load image {FNAME}")
        continue  # Skip the rest of the loop for this iteration
    a, b, c = CC.shape
    #print(CC.shape, c)

    image = imageio.imread(FNAME)
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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)
    imB = transforms.Compose([
        transforms.Resize(256),
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
    RA, RB, RC = get2Dfrom3D(224, 224, r)  # perturbation
    # get the perturbation and the gradient based on the original version
    BRA, BRB, BRC = get2Dfrom3D(224, 224, rB)  # perturbation

    #defence
    L2RD=L2NormValue(RA)
    L2GD = L2NormValue(RB)
    L2BD = L2NormValue(RC)

    LIRD = L_Inf(RA)
    LIGD = L_Inf(RB)
    LIBD = L_Inf(RC)

    #original

    L2RA = L2NormValue(BRA)
    L2GA = L2NormValue(BRB)
    L2BA = L2NormValue(BRC)

    LIRA = L_Inf(BRA)
    LIGA = L_Inf(BRB)
    LIBA = L_Inf(BRC)
    '''
    ValueTime = ['Original ATT,GT', 'Original ATT, ATT', 'On Fake ATT, GT', 'On Fake ATT,ATT', 'On Fake ATT, Def',
                 'ACC', 'ACC_ALL', 'DL2R', 'DL2G', 'DL2B', 'DLIR', 'DLIG', 'DLIB', 'AL2R', 'AL2G', 'AL2B', 'ALIR',
                 'ALIG', 'ALIB']'''
    ValueTime = [label_origB, label_pertB, Originallabel,label_pert, Protected,
                 Acc, AccB, L2RD, L2GD, L2BD, LIRD, LIGD, LIBD,
                 L2RA, L2GA, L2BA, LIRA, LIGA, LIBA]
    writerT.writerow(ValueTime)



print("Final Result:   ",CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def)
if CountTotal > 0:
    percentage = int(CountT * 100 / CountTotal)
else:
    percentage = None  # or handle it in a way that makes sense for your context
ValueTime = [CountT, CountTotal, CountDF_EFF, CountDF_EFF_Def, percentage]
# ValueTime=[CountT,CountTotal,CountDF_EFF,CountDF_EFF_Def,int(CountT*100/CountTotal)]
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
GA,GB,GC=get2Dfrom3D(224,224,TheGradient)   #gradient
RA,RB,RC=get2Dfrom3D(224,224,r)      # perturbation
# get the perturbation and the gradient based on the original version
BGA,BGB,BGC=get2Dfrom3D(224,224,TheGradientB)   #gradient
BRA,BRB,BRC=get2Dfrom3D(224,224,rB)      # perturbation

#
title="Perturbation Compare A, Positive 4"
CVShowCompareFB(RA,BRA,title)
title="Perturbation Compare A, Negtive 4"
CVShowCompareGB(RA,BRA,title)
print("L2")
print(L2NormValue(RA))
print(L2NormValue(BRA))
title="Perturbation Compare A, Positive 0"
CVShowCompareF(RA,BRA,title)
title="Perturbation Compare A, Negtive 0"
CVShowCompareG(RA,BRA,title)

title="Perturbation Compare A, Positive 2"
CVShowCompareFC(RA,BRA,title)
title="Perturbation Compare A, Negtive 2"
CVShowCompareGC(RA,BRA,title)


title="Perturbation Compare B, Positive"
CVShowCompareF(RB,BRB,title)
title="Perturbation Compare B, Negtive"
CVShowCompareG(RB,BRB,title)

title="Perturbation Compare C, Positive"
CVShowCompareF(RC,BRC,title)
title="Perturbation Compare C, Negtive"
CVShowCompareG(RC,BRC,title)

title="Gradient Compare A, Positive"
CVShowCompareF(GA,BGA,title)
title="Gradient Compare A, Negtive"
CVShowCompareG(GA,BGA,title)


title="Gradient Compare B, Positive"
CVShowCompareF(GB,BGB,title)
title="Gradient Compare B, Negtive"
CVShowCompareG(GB,BGB,title)


title="Gradient Compare C, Positive"
CVShowCompareF(GC,BGC,title)
title="Gradient Compare C, Negtive"
CVShowCompareG(GC,BGC,title)



#exit()

#show the heatmap of the perturbation


RA,RB,RC=get2Dfrom3D(224,224,r)
Title="Perturbation Orginal A"
im, cbar = heatmap(RA,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation B"
im, cbar = heatmap(RB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation C"
im, cbar = heatmap(RC,Title,cmap="YlGn", cbarlabel="Perturbation")


RA,RB,RC=get2Dfrom3D(224,224,rB)
Title="Perturbation by Faked GradientA"
im, cbar = heatmap(RA,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation B"
im, cbar = heatmap(RB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Perturbation C"
im, cbar = heatmap(RC,Title,cmap="YlGn", cbarlabel="Perturbation")

GA,GB,GC=HM.get2Dfrom3D(224,224,TheGradient)

Title="Gradient A"
im, cbar = heatmap(GA,Title,cmap="YlGn", cbarlabel="Gradient")
Title="Gradient B"
im, cbar = heatmap(GB,Title,cmap="YlGn", cbarlabel="Perturbation")
Title="Gradient C"
im, cbar = heatmap(GC,Title,cmap="YlGn", cbarlabel="Gradient")
