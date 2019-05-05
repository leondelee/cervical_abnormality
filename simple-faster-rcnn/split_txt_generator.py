#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import os

xml_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/Annotations"
txt_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/ImageSets/Main"
xml_files = os.listdir(xml_base_path)
num = len(xml_files)
train_val = int(0.9*num)
with open(os.path.join(txt_base_path,'trainval.txt'),'w') as f:
    for i in range(train_val):
        xml_file_names = (xml_files[i]).split('.')
        f.write(xml_file_names[0])
        f.write('\n')
    f.close()

with open(os.path.join(txt_base_path,'test.txt'),'w') as f:
    for i in range(train_val,num):
        xml_file_names = (xml_files[i]).split('.')
        f.write(xml_file_names[0])
        f.write('\n')
    f.close()
    


# In[ ]:




