# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/14

import myo
from collections import deque
from threading import Lock
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://kai:password@localhost/db?charset=utf8mb4')
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


def data_process(data):
    """
    数据处理
    跟进div设置间隔，每50行取平均值
    :param df:
    :return:
    """
    div = 50
    averages = int(number_of_samples / div)
    data = np.absolute(data)
    result_data = np.zeros((int(averages), 8))
    for i in range(1, averages + 1):
        result_data[i - 1, :] = np.mean(data[(i - 1) * div:i * div, :], axis=0)
    return result_data


def data_process1(df):
    """

    :param df:
    :return:
    """
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df['gesture'])
    cols = [str(i) for i in range(8)]
    data = df[cols].values
    return data, labels


def train(data, labels):
    """
    训练
    :param df:
    :return:
    """
    labels_num = len(np.unique(labels))

    model = keras.Sequential([
        # Input dimensions means input columns. Here we have 8 columns, one for each sensor
        keras.layers.Dense(8, activation=tf.nn.relu, input_dim=8, kernel_regularizer=regularizers.l2(0.1)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(labels_num, activation=tf.nn.softmax)])

    adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(data, labels, epochs=300, validation_split=0.2, batch_size=16)

    model.save('output/gesture_model.h5')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('output/acc.jpg')

    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('output/loss.jpg')
    plt.show()


def collect_data():
    """

    :return:
    """
    # myo.init()
    myo.init(sdk_path='sdk')
    df = pd.DataFrame()
    gestures = np.loadtxt('data/gesture.csv', dtype='str')
    # gestures = gestures[:3]
    for gesture in gestures:
        print(gesture)
        input("Hold a finger movement:")
        if gesture == 'end':
            break
        data = collect(myo, gesture)
        df = df.append(data)
        # print(gesture)

    df.to_csv('output/gesture_data.csv', index=False)
    df.to_sql('genture_data', engine, if_exists='append', index=False)


def main():
    """

    :return:
    """
    df = pd.read_csv('output/gesture_data.csv')

    # df = df[:5*40]
    data, labels = data_process1(df)
    train(data, labels)


if __name__ == '__main__':
    collect_data()
    # main()
