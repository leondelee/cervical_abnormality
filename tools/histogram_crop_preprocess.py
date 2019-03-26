#!/usr/bin/env python
# coding: utf-8

# In[30]:


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

base_dir = "./vinegar_balanced/vinegar"
save_dir = "./vinegar_hist_crop"
data_set = ["train","validation","test"]
classes = ["1","2","3","cancer"]
for i in range(3):
    for j in range(3):
        work_dir = os.path.join(base_dir,data_set[i])
        work_dir = os.path.join(work_dir,classes[j])
        pic_list = os.listdir(work_dir)
        for pic in pic_list:
            img = cv2.imread(os.path.join(work_dir,str(pic)))
            b,g,r = cv2.split(img) 
#            img_rgb = cv2.merge([r,g,b]) 
#            plt.imshow(img_rgb)
#            plt.show()
            # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            bh = bH[:,200:-200]
            gh = gH[:,200:-200]
            rh = rH[:,200:-200]
            # 合并每一个通道
            #result = cv2.merge((rH, gH, bH))
            result = cv2.merge((bh, gh, rh))
            save_path = os.path.join(os.path.join(save_dir,data_set[i]))
            save_path = os.path.join(os.path.join(save_path,classes[j]))
            if not os.path.exists(save_path):
                os.makedirs(save_path) 
            cv2.imwrite(os.path.join(save_path,pic),result)
#            plt.imshow(result)
#            plt.show()




