# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/17
# UpdatedBy: Youhao Wang

import myo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pickle
import threading
from threading import Lock
import time
from collections import deque
import numpy as np
import argparse
from collect_data import data_process1, collect, engine
from collect_data_csv import *
from model_test import *
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings(action='ignore')


class Listener(myo.DeviceListener):
    def __init__(self, n, clf):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.clf = clf
        self.data = []

    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)

    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        global result
        global vote
        with self.lock:
            self.emg_data_queue.append((event.emg))
            # data = np.abs(list(event.emg))

            data = list(event.emg)
            if len(self.data) < 40:
                """取到40个数据片"""
                self.data.append(data)
            else:
                """这里把传入数据的预处理都做了（绝对值取平均）"""
                data=self.data
                resultnow = str(predict(self.clf, data, mode)[0])
                print(resultnow)
                ##  投票机制
                if result!=resultnow:
                    vote+=1
                    if vote==2:
                        result=resultnow
                        vote=0
                self.data = []

                # if len(list(self.emg_data_queue)) >= self.n:
                #     self.emg_data_queue.clear()
                #     return False


def StackingMethod(X, y):
    '''
    Stacking方法实现分类
    INPUT -> 特征, 分类标签
    Train_test_spilt是随机打乱过的
    而五折交叉验证则保留了时间顺序，要学习到它们的分布，对于RF是困难的。
    '''
    # scaler = StandardScaler() # 标准化转换
    # scaler.fit(X)  # 训练标准化对象
    # traffic_feature= scaler.transform(X)   # 转换数据集
    feature_train, feature_test, target_train, target_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0)

    clf1 = SVC(random_state=1, probability=True)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    clf4 = GradientBoostingClassifier()

    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],
                              use_probas=True,  # 类别概率值作为meta-classfier的输入
                              # average_probas=False,  是否对每一个类别产生的概率值做平均
                              meta_classifier=LogisticRegression())

    sclf.fit(feature_train, target_train)
    # clf2.fit(feature_train, target_train)
    # 模型测试
    predict_results = sclf.predict(feature_test)
    print(accuracy_score(predict_results, target_test))
    conf_mat = confusion_matrix(target_test, predict_results)
    print(conf_mat)
    print(classification_report(target_test, predict_results))

    # 5折交叉验证
    for clf, label in zip([clf1, clf2, clf3, clf4], ['SVC', 'Random Forest', 'naive Bayes', 'Boosting Tree']):
        scores = model_selection.cross_val_score(
            clf, X, y, cv=10, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
              (scores.mean(), scores.std(), label))

    return sclf


def data_process(df, gesture):
    """
    数据处理
    :return:

    """
    data = df[[str(i) for i in range(8)]]
    # data = np.abs(data)
    result_df = pd.DataFrame()
    result_data = []
    for i in range(0, len(data)-20, 20):
        # print(i)
        # result_df = result_df.append(pd.DataFrame(data[i:(i + 40)].mean(axis=0)))
        result_data.append(data[i:(i+40)])
    narray = np.array(result_data)
    result_df = pd.DataFrame.from_records(narray)
    # result_df.newaxis()
    result_df['gesture'] = gesture
    return result_df


def dump_model(path, mode):
    """
    保存模型
    :return:保存随机森林模型pkl
    """
    # path='output/backend/Alldata_wyh.csv'

    df_data = pd.read_csv(path)
    # df_data = pd.read_sql('select * from genture_data', engine)
    df = pd.DataFrame()
    for gesture in df_data.gesture.unique():
        df_sub = df_data[df_data.gesture == gesture]
        df = df.append(data_process(df_sub, gesture))
    # data, labels = data_process1(df)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df['gesture'])
    # cols = [str(i) for i in range(8)]
    lee = df.pop('gesture')
    # data=df[list(range(8))].values

    data = df.values
    with open('models/le.pkl', 'wb') as f:
        pickle.dump(le, f)

    # abs--mean--process //for traditional machine learning 
    ab = np.array(np.abs(data).mean(1).tolist())

    # origin data of shape(-1,40,8) //for end-to-end AI
    dd = np.array(data.tolist())/1
    # data=dd for test
    if mode == 'ml':
        #
        clf = RandomForestClassifier()
        clf.fit(ab, labels)
        print(cross_val_score(clf, ab, labels, cv=10).mean())

        model = StackingMethod(ab, labels)
        print(cross_val_score(model, ab, labels, cv=10).mean())

        # print(cross_val_score(clf, data, labels, cv=5))
        # pickle.dumps('models/randomforest.pkl', clf)
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(clf, f)
    if mode == 'dl':

        Ann_train(dd, labels,5)
        


def load_model(path, mode):
    """
    加载模型
    :return:
    """
    if mode == 'ml':
        f = open(path, 'rb')
        clf = pickle.load(f)
    elif mode == 'dl':
        from tensorflow.keras.models import load_model as lm
        clf = lm('./models/deepl.h5')
    else:
        return
    myo.init(sdk_path='sdk')
    # while True:
    #     df = collect(myo, 'unknown')
    #     data = df[list(range(8))].values
    #     labels = clf.predict(data)

    #     for y in labels:
    #         print(y)
    while True:
        hub = myo.Hub()
        listener = Listener(2000, clf)
        hub.run(listener.on_event, 20000)
        # 修改全局分类结果


def predict(clf, data, mode):
    f = open('models/le.pkl', 'rb')
    le = pickle.load(f)
    if mode == 'ml':
        """机器学习分类器分类"""
        # print(clf.predict([data]))
        data = np.abs(data).mean(axis=0)
        return le.inverse_transform(clf.predict([data]))
    elif mode == 'dl':
        """加载神经网络模型预测"""
        data=np.array(data)
        a = np.argmax(clf.predict(data[np.newaxis, :]))
       # print("here")
        return le.inverse_transform([a])
    else:
        return


def sendResult():
    # 导入模块
    import socket
    global result
    # 创建实例
    client = socket.socket()
    # 访问的服务端的ip和端口
    ip_port = ("192.168.149.1", 7001)
    # 连接主机
    client.connect(ip_port)

    # 定义一个循环，不断发送消息
    while True:
        # 接收主机信息
        data = client.recv(1024)
        # 打印接收的数据
        # 此处byte型数据特指python3.x以上
        # print(data.decode())
        # 输入发送的消息
        # msg_input=input("请输入发送的消息")
        msg_input = result
        # print(result)
        # 消息发送
        client.send(msg_input.encode())
        if msg_input == "exit":
            break
        data = client.recv(1024)
        # print(data.decode())

def portSend():
    #设置全局变量
   
    global result
    from gestures import gestures_command_dict as comdict
    import serial #导入模块
    import time
   
    pastresult='0'
    # #检查端口号
    # port_list = list(serial.tools.list_ports.comports())
    # # print(port_list)
    # if len(port_list) == 0:
    #     print('无可用串口')
    # else:
    #     for i in list(port_list):
    #         print(i[1])
            
    #设置端口
    ser_hand = serial.Serial('COM10', 115200,timeout=2)
    ser_hand.isOpen()
    #指令循环发送
    #TO DO: 指定循环终止条件，循环发送条件，什么时候发送
    # result=['Fist','Two','Five','Thumb_up','Ok']
    if result == 'no':
        time.sleep(3)

    while True:
        print('start')
        if result!=pastresult:
                msg_send = comdict[result]
                ser_hand.write(bytes.fromhex(msg_send) )
                pastresult=result

        else:

            pass
    ser_hand.close()


if __name__ == '__main__':
    global result 
    global mode
    global vote 
    vote=0
    result = 'no'
    mode = 'ml'
    
    
    while True:
        mode_command = input("输入指令：1.机器学习  2.神经网络 ")
        if mode_command == '1':
            mode = 'ml'
        else:
            mode = 'dl'

        command = input("输入指令：1.训练模型  2.读取模型  3.退出")
        
        if command == '1':
            collect_data()

            dump_model('./output/gesture_data.csv', mode)
            
        elif command == '2':
            # threading.Thread(target=sendResult, daemon=True).start()
            threading.Thread(target=portSend, daemon=True).start()
            load_model('models/random_forest.pkl', mode)
            
        elif command == '3':
            break
        else:
            print("请重新输入！")

        # path='./output/gesture_data.csv'

    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('f', choices=['train', 'load'], help='func name')
    args = parser.parse_args()
    if args.f == 'train':
        dump_model('./output/gesture_data.csv')
    if args.f == 'load':
        load_model('models/random_forest.pkl')
   '''
