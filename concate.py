# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:25:39 2021

@author: youhao_wang
"""

#合并数据到Alldata文件夹下的Alldata文件
import os
import pandas as pd
import numpy as np


dir =r'D:\EMG\Code\myo-action-detection-main\output\backend'#设置工作路径

#新建列表，存放文件名（可以忽略，但是为了做的过程能心里有数，先放上）
filename_excel = []

#新建列表，存放每个文件数据框（每一个excel读取后存放在数据框）
frames = []

for root, dirs, files in os.walk(dir):
    print(root,dirs,files)
    for file in files:
        print(os.path.join(root,file))
        filename_excel.append(os.path.join(root,file))
        df = pd.read_csv(os.path.join(root,file)) #excel转换成DataFrame
        frames.append(df)
#打印文件名
print(filename_excel)   
 #合并所有数据
result = pd.concat(frames)    

#查看合并后的数据
result.head()
result.shape

result.to_csv(r'D:\EMG\Code\myo-action-detection-main\output\backend\Alldata_wyh.csv',sep=',',index = False)




dir =r'D:\wechatdata\Year2019'#设置工作路径

#新建列表，存放文件名（可以忽略，但是为了做的过程能心里有数，先放上）
filename_excel = []

#新建列表，存放每个文件数据框（每一个excel读取后存放在数据框）
frames = []

for root, dirs, files in os.walk(dir):
    for file in files:
        #print(os.path.join(root,file))
        filename_excel.append(os.path.join(root,file))
        df = pd.read_csv(os.path.join(root,file)) #excel转换成DataFrame
        frames.append(df)
#打印文件名
print(filename_excel)   
 #合并所有数据
result = pd.concat(frames)    

#查看合并后的数据
result.head()
result.shape

result.to_csv(r'D:\wechatdata\Alldata\Alldata.csv',encoding='utf-8-sig',sep=',',index = False)