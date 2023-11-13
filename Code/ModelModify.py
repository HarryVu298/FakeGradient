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
from HeatMapShow import ShowHeatMap
import AnalysisWeight as AW

def ModifyModel(net2):
    #print(net2)
    TTB = net2.fc.weight
    BiaB = net2.fc.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # 
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======",r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r,c)
    for i in range(r):
        for j in range(c):
            #print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    #ShowHeatMap(WSave)
    Dic=AW.AnalyzeDitriWeight(WSave)
    #=============================================================================================================
    #exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r,c)
    for i in range(r):
        for j in range(c):
            #WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            Scal=2**(-1*Dic[j])
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j]))*Scal

    print("===Generate new weight Done 4========", r, c)
    #======================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.fc = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.fc.weight
    NewBais = net2.fc.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")

    ##exit()
   

    return net2



def ModifyModelScale(net2,Scale):
    #print(net2)
    TTB = net2.fc.weight
    BiaB = net2.fc.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # 
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======",r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * Scale # original
        Flag=1
        if BSaveB[0][i]<0:
            Flag=-1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale*Flag
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r,c)
    for i in range(r):
        for j in range(c):
            #print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    #ShowHeatMap(WSave)
    Dic=AW.AnalyzeDitriWeight(WSave)
    #=============================================================================================================
    #exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r,c)
    for i in range(r):
        for j in range(c):
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale *Flag  #original
            #Scal=2**(-1*Dic[j])
            #WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j]))*Scal
            # WSaveB[i][j] = (WeightMax - WSaveB[i][j])*Scale

    print("===Generate new weight Done 4========", r, c)
    #====================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.fc = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.fc.weight
    NewBais = net2.fc.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")

    

    return net2

def ModifyModelMobNetV2(net2):
    # print(net2)
    TTB = net2.classifier[1].weight
    BiaB = net2.classifier[1].bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # 
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======", r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        # BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r, c)
    for i in range(r):
        for j in range(c):
            # print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    # ShowHeatMap(WSave)
    Dic = AW.AnalyzeDitriWeight(WSave)
    # =============================================================================================================
    # exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r, c)
    for i in range(r):
        for j in range(c):
            # WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            Scal = 2 ** (-1 * Dic[j])
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scal

    print("===Generate new weight Done 4========", r, c)
    # =====================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.classifier[1] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.classifier[1].weight
    NewBais = net2.classifier[1].bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")
    return net2

def ModifyModelDensNet(net2):
    # print(net2)
    TTB = net2.classifier.weight
    BiaB = net2.classifier.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # save biase and weight
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======", r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        # BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r, c)
    for i in range(r):
        for j in range(c):
            # print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    # ShowHeatMap(WSave)
    Dic = AW.AnalyzeDitriWeight(WSave)
    # =============================================================================================================
    # exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r, c)
    for i in range(r):
        for j in range(c):
            # WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            Scal = 2 ** (-1 * Dic[j])
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scal

    print("===Generate new weight Done 4========", r, c)
    # ===================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.classifier = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.classifier.weight
    NewBais = net2.classifier.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")
    return net2

def ModifyModelB(net2):
    #print(net2)
    TTB = net2.fc.weight
    BiaB = net2.fc.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # save biase and weight
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("===================",r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Done 1========", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Done 2========", r,c)
    for i in range(r):
        for j in range(c):
            #print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("===Done 2===Before copy===", r, c)
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("===Done 3========", r,c)
    for i in range(r):
        for j in range(c):
            WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20
    print("===Done 4========", r, c)
    OutL = int(r * 2)
    net2.fc = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.fc.weight
    NewBais = net2.fc.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("======1--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3--------------")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--------------")

   

    return net2

def ModifyModelVGG(net1):
    '''
        VGG modification
    '''
    TT=net1.classifier[0].weight
    print(TT.shape)
    TTA=net1.classifier[3].weight
    print(TTA.shape)
    TTB=net1.classifier[6].weight
    BiaB=net1.classifier[6].bias
    print(BiaB.shape)
    #exit()

    # save biase and weight
    r,c=TTB.shape
    BSave=np.zeros((1,r))
    for i in range(r):
        BSave[0][i]=BiaB[i].clone()
    BSaveB=BSave.copy()
    MaxBias=BSaveB.max()
    for i in range(r):
        BSaveB[0][i]=(MaxBias-BSaveB[0][i])*20
    #exit()

    WSave=np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            WSave[i][j]=TTB[i][j].clone()
    WSaveB=WSave.copy()
    WeightMax=WSaveB.max()

    for i in range(r):
        for j in range(c):
            WSaveB[i][j]=(WeightMax-WSaveB[i][j])*20

    OutL=int(r*2)
    net1.classifier[6]=nn.Linear(in_features=c, out_features=OutL, bias=True)
    NewWeight=net1.classifier[6].weight
    NewBais=net1.classifier[6].bias
    print(r,c)
    print(TTB.shape)
    U=TTB[0][0].clone()
    #TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("======1--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP=WSave[i][j].copy()
                NewWeight[i][j]=TMP

    print("======2--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i+r][j]=TMP
    print("======3--------------")
    with torch.no_grad():
        for i in range(r):
            #BSave[0][i]=BiaB[i].clone()
            TMP=BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i]=TMP
            NewBais[i+r] =TMPB
    print("======4--------------")
    #exit()
    return net1

def ModifyModelVGGScale(net1,Scale):
    '''
        VGG modification
    '''
    TT=net1.classifier[0].weight
    print(TT.shape)
    TTA=net1.classifier[3].weight
    print(TTA.shape)
    TTB=net1.classifier[6].weight
    BiaB=net1.classifier[6].bias
    print(BiaB.shape)
    #exit()

    # save biase and weight
    r,c=TTB.shape
    BSave=np.zeros((1,r))
    for i in range(r):
        BSave[0][i]=BiaB[i].clone()
    BSaveB=BSave.copy()
    MaxBias=BSaveB.max()
    for i in range(r):
        Flag = 1
        if BSaveB[0][i] < 0:
            Flag = -1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
    #exit()

    WSave=np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            WSave[i][j]=TTB[i][j].clone()
    WSaveB=WSave.copy()
    WeightMax=WSaveB.max()

    for i in range(r):
        for j in range(c):
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag  # original
            #WSaveB[i][j]=(WeightMax-WSaveB[i][j])*Scale

    OutL=int(r*2)
    net1.classifier[6]=nn.Linear(in_features=c, out_features=OutL, bias=True)
    NewWeight=net1.classifier[6].weight
    NewBais=net1.classifier[6].bias
    print(r,c)
    print(TTB.shape)
    U=TTB[0][0].clone()
    #TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("======1--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP=WSave[i][j].copy()
                NewWeight[i][j]=TMP

    print("======2--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i+r][j]=TMP
    print("======3--------------")
    with torch.no_grad():
        for i in range(r):
            #BSave[0][i]=BiaB[i].clone()
            TMP=BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i]=TMP
            NewBais[i+r] =TMPB
    print("======4--------------")
    #exit()
    return net1

def ModifyModelDensNetScale(net2,Scale):
    # print(net2)
    TTB = net2.classifier.weight
    BiaB = net2.classifier.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # save biase and weight
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======", r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        Flag = -1
        if BSaveB[0][i] < 0:
            Flag = 1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
        # BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
       #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r, c)
    for i in range(r):
        for j in range(c):
            # print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    # ShowHeatMap(WSave)
    Dic = AW.AnalyzeDitriWeight(WSave)
    # =============================================================================================================
    # exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r, c)
    for i in range(r):
        for j in range(c):
            # WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            #Scal = 2 ** (-1 * Dic[j])
            #WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scal
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag  # original

    print("===Generate new weight Done 4========", r, c)
    # =================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.classifier = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.classifier.weight
    NewBais = net2.classifier.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")
    return net2

def ModifyModelMobNetV2Scale(net2,Scale):
    # print(net2)
    TTB = net2.classifier[1].weight
    BiaB = net2.classifier[1].bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # save biase and weight
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======", r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        # BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
        Flag = 1
        if BSaveB[0][i] < 0:
            Flag = -1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r, c)
    for i in range(r):
        for j in range(c):
            # print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    # ShowHeatMap(WSave)
    Dic = AW.AnalyzeDitriWeight(WSave)
    # =============================================================================================================
    # exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r, c)
    for i in range(r):
        for j in range(c):
            # WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            #Scal = 2 ** (-1 * Dic[j])
            #WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scal
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag  # original

    print("===Generate new weight Done 4========", r, c)
    # ==================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.classifier[1] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.classifier[1].weight
    NewBais = net2.classifier[1].bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")
    return net2


def ModifyModelC(net2):
    ScaleA=0.5
    #print(net2)
    TTB = net2.fc.weight
    BiaB = net2.fc.bias

    # TTB=net1.classifier[6].weight
    print(TTB.shape)

    # save biase and weight
    print("save the bias and weight")
    r, c = TTB.shape
    BSave = np.zeros((1, r))
    print("=======Backup bias=======",r)
    for i in range(r):
        BSave[0][i] = BiaB[i].clone()
    print("===Backup bias Done 1========", r)
    print("=======generate Fake bias=======", r)
    BSaveB = BSave.copy()
    MaxBias = BSaveB.max()
    for i in range(r):
        #BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20   # version 1 original
        BSaveB[0][i] = (MaxBias - BSaveB[0][i]) * 20
    # exit()
    print("===generate Fake bias Done 2========", r)
    WSave = np.zeros((r, c))
    print("===Save the weight========", r,c)
    for i in range(r):
        for j in range(c):
            #print("=======",j,"=++++++++++++++++",i)
            WSave[i][j] = TTB[i][j].clone()
    print("==Save the weight  done=======", r, c)
    #ShowHeatMap(WSave)
    Dic=AW.AnalyzeDitriWeight(WSave)
    #=============================================================================================================
    #exit()
    WSaveB = WSave.copy()
    WeightMax = WSaveB.max()
    print("=====Generate new weight========", r,c)
    for i in range(r):
        for j in range(c):
            #WSaveB[i][j] = (WeightMax - WSaveB[i][j]) * 20  #original
            Scal=2**(-1*Dic[j])
            #WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j]))*Scal
            WSaveB[i][j] = (-WSaveB[i][j])
    print("===Generate new weight Done 4========", r, c)
    #================================
    print("===Generate new layer in the model========", r, c)
    OutL = int(r * 2)
    net2.fc = nn.Linear(in_features=c, out_features=OutL, bias=True)
    # NewWeight=net1.classifier[6].weight
    # NewBais=net1.classifier[6].bias
    NewWeight = net2.fc.weight
    NewBais = net2.fc.bias
    print(r, c)
    print(TTB.shape)
    U = TTB[0][0].clone()
    # TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("========Update new weight and bias========", r, c)
    print("======1-----original weight part---------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSave[i][j].copy()
                NewWeight[i][j] = TMP*ScaleA

    print("======2----fake weight part----------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i + r][j] = TMP
    print("======3-----update fake bias-----")
    with torch.no_grad():
        for i in range(r):
            # BSave[0][i]=BiaB[i].clone()
            TMP = BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBais[i] = TMP
            NewBais[i + r] = TMPB
    print("======4--update new layer done-------")

    return net2