#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import VGG16,Resnet50,Diynet,codingnet
#check model parameters
model = VGG16(input_length=540, input_width=540)
#model = Resnet50()
#model = Diynet(input_length=540, input_width=540)
#model = codingnet(input_length=540, input_width=540)
model.summary()


# In[2]:


import keras
from keras import optimizers
sgd = optimizers.SGD(lr=0.0005, decay=1e-5, momentum=0.9, nesterov=True)
#ada = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[3]:


from data_loader import *
#run model
import tensorflow as tf
import os 
from keras.metrics import top_k_categorical_accuracy

batch_size = 1

train_generator,validation_generator,test_generator = load_image(base_dir='./vinegar_balanced/vinegar',
                                                                 batch_size = batch_size,input_length=540,input_width=540)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#cw = {0:1, 1:2,2:20}


with tf.device("/gpu:1"):
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n/batch_size,  # 2000 images = batch_size * steps
            epochs=15,
            validation_data=validation_generator,
            validation_steps=validation_generator.n/batch_size,  # 1000 images = batch_size * steps
            verbose=1)
#            class_weight = 'auto')
#            class_weight = cw)
#            callbacks=[tensor_board])

model.save('vgg16.h5')
#model.save('Diynet.h5') 
#model.save('resnet50.h5')
scores=model.evaluate_generator(test_generator, steps=test_generator.n/batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





