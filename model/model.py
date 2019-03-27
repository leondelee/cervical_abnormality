#!/usr/bin/env python
# coding: utf-8

# In[4]:


#define  vgg models
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.models import Model
from keras.layers import Dropout
from keras import layers
from keras.layers import Input
from keras.models import Sequential
from keras.initializers import glorot_uniform


def VGG16(input_length=720, input_width=540):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(input_length,input_width,3)))
    model.add(Conv2D(64, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0)))
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(64, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0)))
    model.add(MaxPooling2D((2,2), strides=(2,2))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(128, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(128, 3, activation='relu')) 
    model.add(MaxPooling2D((2,2), strides=(2,2))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(256, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(256, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(256, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(ZeroPadding2D((1,1)))  
    model.add(Conv2D(512, 3, activation='relu', kernel_initializer = glorot_uniform(seed = 0))) 
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten()) 
    model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1024, activation='relu', kernel_initializer = glorot_uniform(seed = 0)))
    model.add(Dropout(0.5))#0.5的概率抛弃一些连接 
    model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1024, activation='relu', kernel_initializer = glorot_uniform(seed = 0)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    return model


#define resnet
def identity_block(X, f, filters, stage, block):
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    F1, F2, F3 = filters
 
    X_shortcut = X
 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
 
    return X

def convolution_block(X, f, filters, stage, block, s=2):
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
 
    X_shortcut = X
 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
 
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut)
 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
 
    return X


def Resnet50(input_shape = (720,540, 3), classes =4):
 
    X_input = Input(input_shape)
 
    X = ZeroPadding2D((3, 3))(X_input)
 
    X = Conv2D(64, (7, 7), strides = (2,2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides = (2,2))(X)
 
    X = convolution_block(X, f = 3, filters = [64,64,256], stage = 2, block = 'a', s = 1)
    X = identity_block(X, 3, [64,64,256], stage=2, block='b')
    X = identity_block(X, 3, [64,64,256], stage=2, block='c')
 
    X = convolution_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')
 
    X = convolution_block(X, f = 3, filters = [256,256,1024], stage = 4, block = 'a', s = 2)
    X = identity_block(X, 3, [256,256,1024], stage=4, block='b')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='c')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='d')    
    X = identity_block(X, 3, [256,256,1024], stage=4, block='e')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='f')
 
    X = convolution_block(X, f = 3, filters = [512,512,2048], stage = 5, block = 'a', s = 2)
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')
 
    X = AveragePooling2D((2, 2), name='avg_pool')(X)
 
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
 
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    
    return model

def Diynet(input_length = 720, input_width = 540):
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(input_length, input_width, 3))
    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)
    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)   
    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)    
    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
#    x = layers.Conv2D(128, 3, activation='relu')(x)
#    x = layers.MaxPooling2D(2)(x)
#    x = layers.Conv2D(256, 3, activation='relu')(x)
#    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)    
    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(1024, activation='relu')(x)   
    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(4, activation='softmax')(x)
    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully 
    # connected layer + sigmoid output layer
    model = Model(img_input, output)
    return model
    
    
def codingnet(input_length =540,input_width=540):
    img_input = layers.Input(shape=(input_length,input_width,3))
    x = layers.Conv2D(32,11,strides=(1,1))(img_input)
    x = layers.Conv2D(32,11,strides=(1,1))(x)
    x = layers.MaxPooling2D(5,strides=(2,2))(x)
    x = layers.Conv2D(64,9,strides=(1,1))(x)
    x = layers.MaxPooling2D(5,strides=(2,2))(x)
    x = layers.Conv2D(128,9,strides=(1,1))(x)
    x = layers.Conv2D(256,8,strides=(1,1))(x)
    x = layers.Flatten()(x)
    output = layers.Dense(4,activation="softmax")(x)
    model = Model(img_input,output)
    return model




