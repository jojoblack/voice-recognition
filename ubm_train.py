'''
brief:
This script trains a UBM using features extracted from
data in train_data_for_UBM

Side effect:
saves the trained UBM in file ubm.txt
'''
import sys,os
import numpy as np
from gmm import *
import pickle



if __name__ == '__main__':
	print __doc__
	getback_params = pickle.load(open('train_data_for_UBM_features.txt','rb'))
	features_M = getback_params[1]
	features_F = getback_params[2]
	del getback_params

	ubms_dir = 'ubms'
	if not os.path.exists(ubms_dir):
		os.mkdir(ubms_dir)

	ubm_M = GMM(n_mix=64, n_dim=12)
	ubm_M.train(features_M)
	ubm_M.write(ubms_dir+r'/ubm_M')

	ubm_F = GMM(n_mix=64, n_dim=12)
	ubm_F.train(features_F)
	ubm_F.write(ubms_dir+r'/ubm_F')



