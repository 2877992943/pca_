import random
import os
import sys
import math
import numpy as np
import copy

from PIL import Image



trainPath = "D://python2.7.6//MachineLearning//pca-1NN//trainface-yr2"
datafile="D://python2.7.6//MachineLearning//pca-1NN//dataVec.txt"
labelfile="D://python2.7.6//MachineLearning//pca-1NN//dataLabel.txt" 
 
     

global classDic;classDic={}
global allLabel;allLabel=[]
global classList;classList=[]
global dataMat
 
dim=32 
dd=dim*dim
######################

def loadData():
    global dim,dd
    global dataMat,classDic,allLabel
    ###################classlist
    m=0
    for filename in os.listdir(trainPath):
        if filename not in classDic:
            classList.append(filename)
            classDic[filename]=m;m+=1 #{tom:0,jim:1...}
    ###########
     
    dataList=[]
    for label in classList:
        for docname in os.listdir(trainPath+'//'+label):
            if docname.endswith('.jpg'):
                im=Image.open(trainPath+'//'+label+'//'+docname).convert('L')
                rsz=im.resize((dim,dim))
                imArr=np.array(rsz,'f')
                imArr=(255.0-imArr)/255.0# (0,255)->(0,1)
                imVec=imArr.flatten();
                dataList.append(imVec)
                ####
                allLabel.append(classDic[label])
                 
    numcase=len(dataList)
    dataMat=np.mat(dataList)
    print 'total image %d with dim %d'%(numcase,dd) #len(array)!=len(list)
    ##############
     
    outPutfile=open(datafile,'w')
    n,d=np.shape(dataMat)
    for i in range(n):
        for j in range(d):
            outPutfile.write(str(dataMat[i,j]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    ######
    outPutfile=open(labelfile,'w')
     
    for label in allLabel:
        outPutfile.write(str(label))
        outPutfile.write(' ')
         
    outPutfile.close()   
            
    
#############
loadData()





 
 

 
    
         
    
    
    







    
    
