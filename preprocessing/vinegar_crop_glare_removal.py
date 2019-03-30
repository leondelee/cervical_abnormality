#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from glareRemoval import *
import cv2
import matplotlib.pyplot as plt

type_name = str(4)
json_base_path = "/home/antoine/antoine/cervical_data/vinegar_center_label/"+type_name
pic_base_path = "/home/antoine/antoine/cervical_data/cervical/vinegar/"+type_name
json_files = os.listdir(json_base_path)
print(json_files)
origin_height = 1440
origin_width = 1080
origin_size = [1440,1080]
cut_height = int(origin_height*0.7/2)
cut_width = int(origin_width*0.7/2)
resize_height = 360
resize_width = 240


# In[ ]:


# process all the images with the same parameters
for i in range(len(json_files)):
    if i%10:
        print('Processing '+str(i)+' image')
    json_file_name = json_files[i]
    json_file_name_s = json_file_name.split('.')
    if(json_file_name_s[1]=='json'):
        pic_name = json_file_name_s[0]+'.jpg'
        picture_path = os.path.join(pic_base_path,pic_name)
        with open(os.path.join(json_base_path,json_files[i]),'r') as load_f:
            load_dict = json.load(load_f)
            center_ratio = load_dict['pre'][0]
            center = [int(center_ratio[0]*origin_size[0]),int(center_ratio[1]*origin_size[1])]

        img = cv2.imread(picture_path)
        b, g, r = cv2.split(img)
        hm=max(0,center[0]-cut_height)
        hp=min(origin_height,center[0]+cut_height)
        wp=min(origin_width,center[1]+cut_width)
        wm=max(0,center[1]-cut_width)
        h = range(hm,hp)
        w = range(wm,wp)
        b = b[w]
        b = b[:,h] 
        g = g[w]
        g = g[:,h] 
        r = r[w]
        r = r[:,h] 
        img = cv2.merge([r,g,b])
        img = cv2.resize(img,(resize_height,resize_width))
        print(pic_name)
        plt.imshow(img)
        plt.show()
        img_new = glareRemove(img,resize_width,resize_height,0.012,3)
        plt.imshow(img_new)
        plt.show()
        plt.imsave('/home/antoine/antoine/cervical_data/vinegar_preprocessed_data/'+type_name+'/'+pic_name,img_new)


# In[ ]:


# to plot images
for i in range(len(json_files)):
    json_file_name = json_files[i]
    json_file_name_s = json_file_name.split('.')
    if(json_file_name_s[1]=='json'):
        pic_name = json_file_name_s[0]+'.jpg'
        picture_path = os.path.join(pic_base_path,pic_name)
        img = cv2.imread(picture_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r,g,b])
        img = cv2.resize(img,(resize_height,resize_width))
        print(pic_name)
        plt.imshow(img)
        plt.show()
        img_new = plt.imread('/home/antoine/antoine/cervical_data/vinegar_preprocessed_data/'+type_name+'/'+pic_name)
        plt.imshow(img_new)
        plt.show()


# In[ ]:


type_name = str(4)
bad_ones = ['0007']
for bad in bad_ones:
    json_file_name = bad+'.json'
    pic_name = (json_file_name.split('.'))[0]+'.jpg'
    picture_path = os.path.join(pic_base_path,pic_name)
    with open(os.path.join(json_base_path,json_file_name),'r') as load_f:
        load_dict = json.load(load_f)
        center_ratio = load_dict['pre'][0]
        center = [int(center_ratio[0]*origin_size[0]),int(center_ratio[1]*origin_size[1])]
        img = cv2.imread(picture_path)
        b, g, r = cv2.split(img)
        hm=max(0,center[0]-cut_height)
        hp=min(origin_height,center[0]+cut_height)
        wp=min(origin_width,center[1]+cut_width)
        wm=max(0,center[1]-cut_width)
        h = range(hm,hp)
        w = range(wm,wp)
        b = b[w]
        b = b[:,h] 
        g = g[w]
        g = g[:,h] 
        r = r[w]
        r = r[:,h] 
        img = cv2.merge([r,g,b])
        img = cv2.resize(img,(resize_height,resize_width))
        print(pic_name)
        plt.imshow(img)
        plt.show()
        img_new = glareRemove(img,resize_width,resize_height,0.005,3,4,2)
        plt.imshow(img_new)
        plt.show()
        plt.imsave('/home/antoine/antoine/cervical_data/vinegar_preprocessed_data/'+type_name+'/'+pic_name,img_new)

