# -*- coding: utf-8 -*-
'''
brief:
This script adapts the UBM to each speaker

side effect:
Saves speakers' models in adaption/
'''
import sys,os
import pickle
import numpy as np
import mywave
from gmm import *
from MFCC import *


if __name__ == '__main__':
	print __doc__

	ubms_dir = 'ubms'
	speaker_model_dir = 'adaption'
	if not os.path.exists(speaker_model_dir):
		os.mkdir(speaker_model_dir)

	train_data_dir = 'train_data'
	train_data = os.listdir(train_data_dir)
	wav = mywave.mywave()
	for train_wav in train_data:
		print train_wav
		wave_data = wav.WaveRead(train_data_dir+r'/'+train_wav)
		MFCC_obj = MFCC(40,12,300,3400,0.97,16000,50,0.0256,256)
		MFCC_coef = MFCC_obj.sig2s2mfc(wave_data)
		adapted_gmm = GMM()
		if train_wav[-5] == 'M':
			adapted_gmm.read(ubms_dir+r'/ubm_M')
		elif train_wav[-5] == 'F':
			adapted_gmm.read(ubms_dir+r'/ubm_F')
		else:
			print 'train_wav name unexpected'

		adapted_gmm.adapt(MFCC_coef)
		adapted_gmm.write(speaker_model_dir+r'/'+train_wav)


