# -*- coding: utf-8 -*-
'''
brief:
This script extracts MFCC features from the training data

side effect:
MFCC features are saved in file features.txt
'''

import sys,os
from MFCC import *
import numpy as np
import pickle
from mywave import *


if __name__ == '__main__':
	print __doc__

	#  extract features from UBM data
	ubm_dir = 'train_data_for_UBM'
	ubm_data_dirs = os.listdir(ubm_dir)
	
	sig = np.array([])
	features_M = np.ndarray(shape = (0,12), dtype = 'float64')
	features_F = np.ndarray(shape = (0,12), dtype = 'float64')
	features = np.ndarray(shape = (0,12), dtype = 'float64')
	wav = mywave()

	for ubm_data_dir in ubm_data_dirs:
		print ubm_data_dir
		sig = wav.WaveRead(ubm_dir+r'/'+ubm_data_dir)
		MFCC_obj = MFCC(40,12,300,3400,0.97,16000,50,0.0256,256)
		MFCC_coef = MFCC_obj.sig2s2mfc(sig)
		features = np.vstack((features,MFCC_coef))
		if ubm_data_dir[-5] == 'M':
			features_M = np.vstack((features,MFCC_coef))
		elif ubm_data_dir[-5] == 'F':
			features_F = np.vstack((features,MFCC_coef))

	pickle.dump([features,features_M,features_F],open(ubm_dir+r'_features.txt','wb'))
