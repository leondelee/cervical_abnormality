#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def intersection(m,i,j,Mask,img,length,width,glare,num):
    mask = Mask
    img_new = img
    ind = 0
    r = []
    g = []
    b = []
    if Mask[max(0,i-1),max(0,j-1)]==0:
        r.append(img[max(0,i-1),max(0,j-1),0])
        g.append(img[max(0,i-1),max(0,j-1),1])
        b.append(img[max(0,i-1),max(0,j-1),2])
        ind = ind+1
    if Mask[i,max(0,j-1)]==0:
        r.append(img[i,max(0,j-1),0])
        g.append(img[i,max(0,j-1),1])
        b.append(img[i,max(0,j-1),2])
        ind = ind+1
    if Mask[min(length-1,i+1),max(0,j-1)]==0:
        r.append(img[min(length-1,i+1),max(0,j-1),0])
        g.append(img[min(length-1,i+1),max(0,j-1),1])
        b.append(img[min(length-1,i+1),max(0,j-1),2])
        ind = ind+1
    if Mask[max(0,i-1),j]==0:
        r.append(img[max(0,i-1),j,0])
        g.append(img[max(0,i-1),j,1])
        b.append(img[max(0,i-1),j,2])
        ind = ind+1
    if Mask[min(length-1,i+1),j]==0:
        r.append(img[min(length-1,i+1),j,0])
        g.append(img[min(length-1,i+1),j,1])
        b.append(img[min(length-1,i+1),j,2])
        ind = ind+1
    if Mask[max(0,i-1),min(width-1,j+1)]==0:
        r.append(img[max(0,i-1),min(width-1,j+1),0])
        g.append(img[max(0,i-1),min(width-1,j+1),1])
        b.append(img[max(0,i-1),min(width-1,j+1),2])
        ind = ind+1
    if Mask[i,min(width-1,j+1)]==0:
        r.append(img[i,min(width-1,j+1),0])
        g.append(img[i,min(width-1,j+1),1])
        b.append(img[i,min(width-1,j+1),2])
        ind = ind+1
    if Mask[min(length-1,i+1),min(width-1,j+1)]==0:
        r.append(img[min(length-1,i+1),min(width-1,j+1),0])
        g.append(img[min(length-1,i+1),min(width-1,j+1),1])
        b.append(img[min(length-1,i+1),min(width-1,j+1),2])
        ind = ind+1
    if ind>num:
        #print(ind)
        mask[i,j]=0
        #glare.remove([i,j])
        index = glare.index([i,j])
        glare[index] = [0,0]
        r = np.array(r) 
        g = np.array(g) 
        b = np.array(b) 
        img_new[i,j,0]=r.mean()
        img_new[i,j,1]=g.mean()
        img_new[i,j,2]=b.mean()
    return mask,img_new,glare


# In[4]:


#time_start=time.time()
#shape:height width channel
#resize width height
#for num in range(1,10):
#    picture_path = '000'+str(num)+'.jpg'
def glareRemove(img,width,height,alpha,num,kernel_size1=2,kernel_size2=1):
    min_ = img.min(2)
    max_ = img.max(2)
    mean_ = img.mean(2)
    I = min_/255
    S = min_/max_
    G = abs(I-S)*255


    for i in range(width):
        for j in range(height):
            if G[i,j]>alpha*255: #hyperparameter
                G[i,j]=0
            else:
                G[i,j]=255
    #print(G.shape)
    #print(G)            
    # plt.imshow(G)
    # plt.show()

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size1, kernel_size1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size2, kernel_size2))

    G_dilate = cv2.dilate(G,kernel1)
    G_erosion = cv2.erode(G_dilate, kernel2)


    Mask = G_erosion
    Mask_temp = Mask
    #print(Mask)
    img_new = img
    #ind1 = 0
    glare = []
    for alpha in range(width):
        for beta in range(height):
            if Mask[alpha,beta]==255:
                glare.append([alpha,beta])
    #            ind1=ind1+1

    glare_left = len(glare)
    #M = Mask.max()
    while glare_left>0:
        glare_left = len(glare)
#        print(glare_left)
        for m in range(glare_left-1,-1,-1):
            temp = glare[m]
            p = temp[0]
            q = temp[1]
            Mask,img_new,glare = intersection(m,p,q,Mask_temp,img_new,width,height,glare,num)
        while [0,0] in glare:
            glare.remove([0,0])
        glare_left = len(glare)
#        print('left: ',glare_left)
        Mask_temp = Mask
    return img_new





