#!/usr/bin/env python
# coding: utf-8

# In[3]:


import shutil,os
img_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/"
xml_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/Annotations"
xml_files = os.listdir(xml_base_path)
num = len(xml_files)
for i in range(num):
    xml_file_names = (xml_files[i]).split('.')
    if xml_file_names[1] == 'xml':
        pic_name = xml_file_names[0] + '.jpg'
#复制单个文件
        shutil.copy(img_base_path+'Images/'+pic_name,img_base_path+'JPEGImages/')


# In[ ]:




