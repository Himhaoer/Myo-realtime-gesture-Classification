# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:41:08 2021

@author: youhao_wang
"""

# %matplotlib qt
import myo
from collections import deque
from threading import Lock
import time
import numpy as np
import pandas as pd
from gestures import gestures
from sklearn import preprocessing
import keras
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, optimizers, Input, Model
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations,callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from tensorflow.keras.optimizers import SGD as sgd #, Adam
from keras.layers import BatchNormalization,AveragePooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
import copy
# engine = sqlalchemy.create_engine('mysql+pymysql://kai:password@localhost/db?charset=utf8mb4')
number_of_samples = 2000



def prepare_data(data,labels):
    '''
    prepare data for cnn
    '''
    sc = MinMaxScaler(feature_range = (0, 1))
    # data origin (xxx,65)
    
    # data[:,:] = sc.fit_transform(data[:,:])
    np.random.shuffle(data)
    
    #spilt data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.reshape(-1,320)
    x_test = x_test.reshape(-1,320)
    
    return x_train, x_test, y_train, y_test
    

def callbacks(Log,Dir):
  import tensorflow as tf
  import os

  Filepath = Path
  logdir = os.path.join(Filepath, Dir)
  
  callbacks_list = [
                    # tf.keras.callbacks.TensorBoard(
                    # log_dir=logdir,                 #  tensorboard log path      
                    # histogram_freq=1,),
                    # tf.keras.callbacks.EarlyStopping(   # stop if not improving
                    # monitor='val_accuracy',patience=10,),           # monitor validation accuracy
                    tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',factor=0.5, cooldown=10,    # lr ko .1 se multiply kerdo (kam kerdo)
                    patience=20,),                # reduce the lrate if val loss stop improving
                    # tf.keras.callbacks.ModelCheckpoint(
                    # filepath= Filepath,             # save model path
                    # monitor='val_loss',             # only save best weights
                    # save_best_only=True,)
                    ]
  return callbacks_list
Path = "model1"
Dir = "my_log_dir"   
Call_B_Fun = callbacks(Path,Dir)


def plot(loss,val_loss,acc,val_acc,history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

def lstm_cnn_model():
    '''
    """Conv1d + Lstm model"""
    '''
    # 11500,178
    
    
    model = Sequential()
    '''思考用什么方式转换为的图数据输入'''
    model.add(layers.Conv1D(32, 3, strides=2,input_shape=(41, 8), use_bias=False))
    model.add(layers.ReLU())
    model.add(layers.Conv1D(32, 3))
    model.add(layers.Conv1D(32, 3, strides=2))
    model.add(layers.ReLU())
    model.add(layers.Conv1D(32, 3))
    model.add(layers.Conv1D(32, 3, strides=2))  # [None, 54, 64]
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32, dropout=0.5, return_sequences=True))
    model.add(layers.LSTM(32, dropout=0.5, return_sequences=True))
    model.add(layers.LSTM(16))
    model.add(layers.Dense(5, activation="softmax"))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer = keras.optimizers.Adam(learning_rate=5e-3),
                  metrics = ["accuracy"])
    
    model.summary()
    
    return model

def embed_conv_model():
    '''
    """ Embedding + Conv1D  """
    Returns
    -------
    model_conv : TYPE
        DESCRIPTION.
    output1 : TYPE
        DESCRIPTION.

    '''
    conv_inputs = layers.Input(shape = (8,))
    
    embed = layers.Embedding (input_dim = 64, 
                  output_dim = 7,
                  embeddings_regularizer='l2')(conv_inputs)
    embed = layers.Conv1D(12,1,activation = 'relu')(embed)        
    embed = layers.Flatten()(embed)
    hidden = layers.Dropout(0.3)(embed)

    hidden = tfa.layers.WeightNormalization(
        layers.Dense(
        units=32,
        activation ='selu',
        kernel_initializer = "lecun_normal"))(hidden)

    output = layers.Dropout(0.3)(layers.Concatenate()([embed, hidden]))
    output = tfa.layers.WeightNormalization(
    layers.Dense(
        units = 32,
        activation='relu',
        kernel_initializer = "lecun_normal"))(output) 
    output = layers.Dropout(0.4)(layers.Concatenate()([embed, hidden, output]))
    output1 = tfa.layers.WeightNormalization(
    layers.Dense(
        units = 32, 
        activation = 'relu',
        kernel_initializer = "lecun_normal"))(output)

    conv_outputs = layers.Dense(
        units = 5, 
        activation ='softmax',
        kernel_initializer ="lecun_normal")(output1)

    model_conv = Model(conv_inputs,conv_outputs)
    model_conv.compile(loss='categorical_crossentropy', 
                  optimizer = keras.optimizers.Adam(learning_rate=2e-4),
                  metrics = ["accuracy"])
    model_conv.summary()
    
    return model_conv, output1

def lstm_dense_model():
    '''
    """lstm模型加linear堆叠"""
    '''
    return 


def resNet():


    K.set_learning_phase(1)
    def abs_backend(inputs):
        return K.abs(inputs)
    
    def expand_dim_backend(inputs):
        return K.expand_dims(K.expand_dims(inputs,1),1)
    
    def sign_backend(inputs):
        return K.sign(inputs)
    
    def pad_backend(inputs, in_channels, out_channels):
        pad_dim = (out_channels - in_channels)//2
        inputs = K.expand_dims(inputs,-1)
        inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(pad_dim,pad_dim)), 'channels_last')
        return K.squeeze(inputs, -1)

    def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
        residual = incoming
        in_channels = incoming.get_shape().as_list()[-1]
        
        for i in range(nb_blocks):
            
            identity = residual
            
            if not downsample:
                downsample_strides = 1
            
            residual = BatchNormalization()(residual)
            residual = Activation('relu')(residual)
            residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides), 
                              padding='same', kernel_initializer='he_normal', 
                              kernel_regularizer=l2(1e-4))(residual)
            
            residual = BatchNormalization()(residual)
            residual = Activation('relu')(residual)
            residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                              kernel_regularizer=l2(1e-4))(residual)
            
            # Calculate global means
            residual_abs = Lambda(abs_backend)(residual)
            abs_mean = GlobalAveragePooling2D()(residual_abs)
            
            # Calculate scaling coefficients
            scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                           kernel_regularizer=l2(1e-4))(abs_mean)
            scales = BatchNormalization()(scales)
            scales = Activation('relu')(scales)
            scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
            scales = Lambda(expand_dim_backend)(scales)
            
            # Calculate thresholds
            thres = keras.layers.multiply([abs_mean, scales])
            
            # Soft thresholding
            sub = keras.layers.subtract([residual_abs, thres])
            zeros = keras.layers.subtract([sub, sub])
            n_sub = keras.layers.maximum([sub, zeros])
            residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
            
            # Downsampling (it is important to use the pooL-size of (1, 1))
            if downsample_strides > 1:
                identity = AveragePooling2D(pool_size=(1,1), strides=(2,2))(identity)
                
            # Zero_padding to match channels (it is important to use zero padding rather than 1by1 convolution)
            if in_channels != out_channels:
                identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
            
            residual = keras.layers.add([residual, identity])
    
        return residual
    inputs = Input(shape=(40,8,1))
    net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    net = residual_shrinkage_block(net, 1, 8, downsample=True)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = GlobalAveragePooling2D()(net)
    outputs = Dense(5, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model
    
    
def dense_model():
    '''
    """Dense**"""
    
    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    input_tensor = Input(shape=(8,))
    x = layers.Dense(1024, activation='relu')(input_tensor)
    y = layers.Dense(512, activation='relu')(x)
    z = layers.Dense(128, activation='relu')(y) # acyclic graghs of layers
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    output_tensor = layers.Dense(5, activation='softmax')(z)
    
    model = Model(input_tensor, output_tensor)
    
    #SGD #RMSprop #Adam #Adadelta #Adagrad ##Adamax ###Nadam #Ftrl
    opt = optimizers.Nadam(lr=3e-3)
    model.compile(optimizer = opt, 
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])
    
    model.summary()
  
    return model

def addFeature(x_train):
    x_tra=np.append(x_train,np.abs(x_train).mean(axis=1).reshape(-1,1,8),axis=1)
    return x_tra



def Ann_train(data,labels,choosemodel=1):  
    """
    choosemodel=1:embed_conv,2:dense,3:cnn,4:lstm_cnn,5:resnet
    train
    """
    # 测试集/训练集分割 
    # 时序shuffle很讲究-- gap spilt or timeseries spilt
    # 标签独热编码
    labels = tf.keras.utils.to_categorical(labels)
    spilt_rate=0.1
    # data=dd
    # x_valid,y_valid=data[(int)(spilt_rate*len(data)):],labels[(int)(spilt_rate*len(labels)):]
    # x_train, x_test, y_train, y_test = train_test_split(data[:(int)(spilt_rate*len(data))], labels[:(int)(spilt_rate*len(labels))],
    #                                                     test_size=0.2, random_state=1)
    
    
    '''
    时间序列专用的数据划分方式
    '''
    from sklearn.model_selection import TimeSeriesSplit

    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4, 5, 6])
    tscv = TimeSeriesSplit(n_splits=3)
    print(tscv)  
    TimeSeriesSplit(n_splits=3)
    for train, test in tscv.split(X):
         print("%s %s" % (train, test))
         
         
    ##把绝对平均值接入，（，41，8）传入
    x_tra = addFeature(x_train)
    x_te = addFeature(x_test)
    # x_va = addFeature(x_valid)
    
    '''
    # 数据标准化
    # mean = x_train.mean(axis=0)
    # std = x_train.std(axis=0)
    # x_train -= mean
    # x_train /= std
    # x_test -= mean
    # x_test /= std
    '''
    # sc=MinMaxScaler(feature_range=(0,10))
    # sc2=StandardScaler()
    # def scaler(data):
    #     result=copy.deepcopy(data)/1
    #     for i in range (data.shape[0]):
    #         result[i]=sc.fit_transform(data[i,:])
    #     return result
    # x_sc_train=scaler(x_train)
    # x_sc_test=scaler(x_test)
    # x_sc_valid=scaler(x_valid)
  
    batch_size = 16            
    epochs = 120   

    
    if choosemodel == 1:
        '''embed_conv_model'''
        model, output1 = embed_conv_model()
        nowtrain=abs(x_train).mean(1)#.tolist().astype('float64')
        nowtest=abs(x_test).mean(1)
        history = model.fit(nowtrain, y_train,
                    batch_size = batch_size, epochs = epochs,
                    validation_data=(nowtest, y_test),
                    callbacks= Call_B_Fun,
                    verbose=0)
        
    elif choosemodel == 2:
        '''dense_model'''
        model=dense_model()
        # nowtrain = x_train.reshape(-1,320)
        # nowtest = x_test.reshape(-1,320)
        nowtrain=abs(x_train).mean(1)#.tolist().astype('float64')
        nowtest=abs(x_test).mean(1)
        # nowvalid=abs(x_valid).mean(1)
        history = model.fit(nowtrain, y_train,
                            batch_size=batch_size, epochs = epochs,validation_data=(nowtest, y_test),#validation_split = 0.2,
                             callbacks=tf.keras.callbacks.ReduceLROnPlateau())
   
    elif choosemodel == 3:
        '''cnn_model'''
        pass
    
    elif choosemodel ==4:
        '''lstm_cnn_model'''
        
        # x_train, x_test, y_train, y_test = prepare_data(data, labels)
        model=lstm_cnn_model()
        ## 可否直接fit 不要callback
        history = model.fit(x_tra, y_train,
                            batch_size=batch_size, epochs = epochs,validation_data=(x_te, y_test),#validation_split = 0.2,
                             callbacks=Call_B_Fun)
 
    elif choosemodel ==5:
        '''resNet_model'''#https://www.cnblogs.com/uizhi/p/12239690.html
        model=resNet()
        # model = tf.keras.models.load_model('./models/test.h5')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size, epochs = epochs,validation_data=(x_test, y_test),
                            #validation_split = 0.2,
                            callbacks=Call_B_Fun)
        
        model.save('./models/deepl.h5')
    
    
    '''model_eval & record'''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plot(loss,val_loss,acc,val_acc,history)
   
    # if choosemodel>3:
    #     model.evaluate(x_te, y_test)
    #     model.evaluate(nowvalid, y_valid)
       
    # np.argmax(model.predict(x_test[:1].shape)[0])
    # np.argmax(y_test[196])
    
    # model.save('./models/deepl.h5')
    

# if __name__ == '__main__':
    # collect_data()
    # main()
