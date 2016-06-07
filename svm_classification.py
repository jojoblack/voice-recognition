
import pickle
import numpy as np
import os
#os.chdir('C:\Users\yqRubick\Anaconda2\Library\libsvm-3.21\python')

os.chdir('C:\Users\yqRubick\Desktop\proj\libsvm-3.21\python')
from svmutil import *

#这一段取数据多次运行时可以注释掉省时间

data = pickle.load(open('C:\Users\yqRubick\Desktop\proj\Vec\svm_data','rb'))


x = []
y = []

for i in range(0,np.size(data)/2):
   y.append(data[i][0])
   x.append(data[i][1])


test_data= pickle.load(open('C:\Users\yqRubick\Desktop\proj\Vec\mytest_data','rb'))


x_t = []
y_t = []

for i in range(0,np.size(test_data)/2):
   y_t.append(test_data[i][0])
   x_t.append(test_data[i][1])

m = svm_train(y,x,'-t 0 -c 19.5')  # 20
#m = svm_load_model('predict.model')

p_lable , p_acc , p_val = svm_predict(y_t,x_t,m)

svm_save_model('predict.model',m)
