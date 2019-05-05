#!/usr/bin/env python
# coding: utf-8

# In[7]:


import json
import os

json_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/labels"
xml_base_path = "/home/antoine/antoine/cervical_data/cervix_detection_dataset/Annotations"
json_files = os.listdir(json_base_path)


# In[8]:


from xml.dom.minidom import Document
# import xml.dom.minidom

class XmlMaker:
    def __init__(self,jsonpath,jsonname,xmlpath):
        self.jsonPath = jsonpath
        self.xmlPath = xmlpath
        self.txtList = []
        self.jsonName = (jsonname.split('.'))[0]
        self.jsonfile = []
        self.num = 0


    def readjson(self):
        with open(os.path.join(self.jsonPath,self.jsonName+'.json'),'r') as load_f:
            json_file = json.load(load_f)
            self.jsonfile = json_file
        self.num = len(self.jsonfile['class'])

    def makexml(self):
        doc = Document()
        annotation = doc.createElement("annotation")
        doc.appendChild(annotation)
        folder = doc.createElement("folder")
        fol_nd = doc.createTextNode("cervix")
        folder.appendChild(fol_nd)
        annotation.appendChild(folder)
        
        filename = doc.createElement("filename")
        jname = doc.createTextNode(self.jsonName)
        filename.appendChild(jname)
        annotation.appendChild(filename)
        
        size = doc.createElement("size")
        annotation.appendChild(size)
        
        height = doc.createElement("height")
        height_nd = doc.createTextNode('1440')
        height.appendChild(height_nd)
        size.appendChild(height)
        
        width = doc.createElement("width")
        width_nd = doc.createTextNode('1080')
        width.appendChild(width_nd)
        size.appendChild(width)
        
        depth = doc.createElement("depth")
        depth_nd = doc.createTextNode('3')
        depth.appendChild(depth_nd)
        size.appendChild(depth)
        
        jsonfile = self.jsonfile
        if self.num>0:
            for i in range(self.num):
                obj = doc.createElement('object')
                annotation.appendChild(obj)
                cls = jsonfile['class'][i][4]
                cls_nd = doc.createElement('name')
                class_node = doc.createTextNode(str(cls))
                cls_nd.appendChild(class_node)
                obj.appendChild(cls_nd)
                
                
                diff_nd = doc.createElement('difficult')
                diff_node = doc.createTextNode('0')
                diff_nd.appendChild(diff_node)
                obj.appendChild(diff_nd)
                
                bnd_nd = doc.createElement('bndbox')
                obj.appendChild(bnd_nd)
                
                coor_nd = doc.createElement('xmin')
                coor_node = doc.createTextNode(str(int(int(jsonfile['class'][i][0])/0.8)))
                coor_nd.appendChild(coor_node)
                bnd_nd.appendChild(coor_nd)
                
                coor_nd = doc.createElement('ymin')
                coor_node = doc.createTextNode(str(int(int(jsonfile['class'][i][1])/0.8)))
                coor_nd.appendChild(coor_node)
                bnd_nd.appendChild(coor_nd)
                
                coor_nd = doc.createElement('xmax')
                coor_node = doc.createTextNode(str(int(int(jsonfile['class'][i][2])/0.8)))
                coor_nd.appendChild(coor_node)
                bnd_nd.appendChild(coor_nd)
                
                coor_nd = doc.createElement('ymax')
                coor_node = doc.createTextNode(str(int(int(jsonfile['class'][i][3])/0.8)))
                coor_nd.appendChild(coor_node)
                bnd_nd.appendChild(coor_nd)
                
            f = open(self.xmlPath, 'w')
            doc.writexml(f, indent='\t', newl='\n', addindent='\t')
            f.close()


# In[9]:


#if __name__ == "__main__":
for i in range(len(json_files)):
    if i%10==0:
        print('Processing '+str(i)+' jsonfile')
    json_file_name = json_files[i]
    json_file_name_s = json_file_name.split('.')
    if(json_file_name_s[1]=='json'):
        xml_name = json_file_name_s[0]+'.xml'
        xml_path = os.path.join(xml_base_path,xml_name)
    read =XmlMaker(json_base_path,json_file_name,xml_path)
    read.readjson()
    read.makexml()


# In[ ]:




