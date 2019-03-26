#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#define paths
import os
#do data augmentation and define generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_image(base_dir,input_length,input_width,batch_size = 1):
    # base_dir = './vinegar'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir,'test')
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255,
    #                                  featurewise_center=True,
       samplewise_center=True,
    #    featurewise_std_normalization = True,
        samplewise_std_normalization = True,
#        zca_whitening = False,
#        rotation_range = 30,
#        shear_range = 0.2,
#        zoom_range = 0.2,
#        horizontal_flip= True,
 #       vertical_flip = True)
                                      )
    validation_datagen = ImageDataGenerator(rescale=1./255,
    #                                 featurewise_center=True,
        samplewise_center=True,
    #    featurewise_std_normalization = True,
        samplewise_std_normalization = True,
#        zca_whitening = False,
#        rotation_range = 30,
#        shear_range = 0.2,
#        zoom_range = 0.2,
#        horizontal_flip= True,
 #       vertical_flip = True)
                                      )
    test_datagen = ImageDataGenerator(rescale=1./255,
    #                                 featurewise_center=True,
        samplewise_center=True,
     #   featurewise_std_normalization = True,
        samplewise_std_normalization = True,
#        zca_whitening = False,
#        rotation_range = 30,
#        shear_range = 0.2,
#        zoom_range = 0.2,
#        horizontal_flip= True,
 #       vertical_flip = True)
                                      )
    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=(input_length,input_width),  # All images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)
    
    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(input_length,input_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(input_length,input_width),
            batch_size=batch_size,
            class_mode='categorical')
    return train_generator,validation_generator,test_generator
    


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize():
    
# Directory with our training pictures
    train_type2 = os.path.join(train_dir, '2')

# Directory with our training pictures
    train_type3 = os.path.join(train_dir, '3')

    train_type_cancer = os.path.join(train_dir,'cancer')

# Directory with our validation pictures
    validation_type2 = os.path.join(validation_dir, '2')

# Directory with our validation pictures
    validation_type3 = os.path.join(validation_dir, '3')

    validation_type_cancer = os.path.join(validation_dir,'cancer')

# Directory with our test pictures
    test_type2 = os.path.join(test_dir, '2')

# Directory with our test pictures
    test_type3 = os.path.join(test_dir, '3')

    test_type_cancer = os.path.join(test_dir,'cancer')
    
    # show data examples, no need for training
# Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_type2_pix = [os.path.join(train_type2, fname) 
                for fname in train_type2_fnames[pic_index-8:pic_index]]
    next_type3_pix = [os.path.join(train_type3, fname) 
                for fname in train_type3_fnames[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_type2_pix+next_type3_pix):
      # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


# In[4]:


import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_image_svm(base_dir = "./vinegar_hist_crop",input_size = 270):
    data_set = ["train","validation","test"]
    classes = ["1","2","3","cancer"]
    data = []
    label = []
    for i in range(3):
        for j in range(4):
            work_dir = os.path.join(base_dir,data_set[i])
            work_dir = os.path.join(work_dir,classes[j])
            pic_list = os.listdir(work_dir)
            for pic in pic_list:
                img = Image.open(os.path.join(work_dir,str(pic)))  #PIL 的 open() 函数用于创建 PIL 图像对象
                img = img.resize((input_size, input_size), Image.ANTIALIAS)
                arr = np.asarray(img,dtype='float32')  #Convert the input to an array
                arr = arr.flatten()
                data.append(arr)
                label.append(i)
    data = np.array(data)
    label = np.array(label)
    return data,label
            




