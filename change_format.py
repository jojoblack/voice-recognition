# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf8')

manu = os.listdir('C:\Users\yqRubick\Desktop\proj\supervector')

vect_to_store = []
for i in range(2,len(manu)-2):  #len(manu)-2
    f = os.listdir('C:\Users\yqRubick\Desktop\proj\supervector_test\\'+manu[i])
    for j in range(0,len(f)):    #每个人提取5个特征试试,提取特征是0~5，采集测试数据是5~8
        pk_file = open('C:\Users\yqRubick\Desktop\proj\supervector\\'+manu[i]+'\\'+f[j],'r')
        tpvect = pickle.load(pk_file)
        
        temp = []
        for k in range(0,128):
            temp = np.append(temp,tpvect[(k-1)*12:k*12])
        newvect = {}
        for k in range(0,1536):
           newvect[k+1]=temp[k]
        new = [i-1,newvect]
        vect_to_store.append(new)


pickle.dump(vect_to_store,open('C:\Users\yqRubick\Desktop\proj\Vec\mytest_data','wb'))   #文件名一个是svm_data,一个是mytest_data
