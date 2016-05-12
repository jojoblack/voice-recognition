# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import numpy
from scipy.fftpack import dct
from scipy.io import wavfile

import os
os.chdir('C:\Users\yqRubick\Anaconda2\Library\libsvm-3.21\python')
from svmutil import *
os.chdir('C:\Users\yqRubick\Desktop')
from MFCC import *


#MFCCµ÷ÓÃ·½Ê½
sample_rate, signal = wavfile.read("test.wav")

MFCC_obj = MFCC(40,12,300,3000,0.97,sample_rate,100,0.0256,256)   #ÆäÖÐ100ÎªÖ¡³¤£¬¾ö¶¨MFCCµÄÎ¬Êý
MFCC_coef = MFCC_obj.sig2s2mfc(signal)

#SVMµ÷ÓÃ·½Ê½
os.chdir('C:\Users\yqRubick\Anaconda2\Library\libsvm-3.21\python')
y,x = svm_read_problem('../heart_scale')
m = svm_train(y[:200],x[:200],'-c 4')
p_lable , p_acc , p_val = svm_predict(y[200:],x[200:],m)