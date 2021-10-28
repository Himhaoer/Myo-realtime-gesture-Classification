# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/25

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
from sklearn.preprocessing import MinMaxScaler
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from tensorflow.keras.optimizers import SGD as sgd #, Adam
from model_test import *
# engine = sqlalchemy.create_engine('mysql+pymysql://kai:password@localhost/db?charset=utf8mb4')
number_of_samples = 2000


class Listener(myo.DeviceListener):
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.data_array = []

    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)

    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.emg))
            if len(list(self.emg_data_queue)) >= self.n:
                self.data_array.append(list(self.emg_data_queue))
                self.emg_data_queue.clear()
                return False



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
    

def collect(myo, gesture):
    """
    收集数据
    :param myo:
    :return:
    """
    hub = myo.Hub()
    listener = Listener(number_of_samples)
    hub.run(listener.on_event, 20000)
    data_set = np.array((listener.data_array[0]))
    # data_set = data_process(data_set)

    df = pd.DataFrame(data_set)
    df['gesture'] = gesture
    return df


def collect_data():
    """
    :return:
    """
    # myo.init()
    myo.init(sdk_path='sdk')
    df = pd.DataFrame()
  #  gestures = np.loadtxt('data/gesture.csv', dtype='str')
    # gestures = gestures[:3]
    for i in range(1):
        for gesture in gestures:
            print(gesture)
            input("Hold a finger movement:")
            
            if gesture == 'end':
                break
            data = collect(myo, gesture)
            df = df.append(data)
            # print(gesture)

    df.to_csv('output/backend/gesture_data_%d.csv'%(int(time.time())), index=False)
    df.to_csv('output/gesture_data.csv', index=False)

    

# if __name__ == '__main__':
    # collect_data()
    # main()
