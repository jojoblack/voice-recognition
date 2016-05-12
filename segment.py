'''
Brief:
This file segment all the .wav files. First we extract the voice 
activity part in the .wav file, then split it into two parts, the train data part 
and the test data part.
'''

import sys,os
import numpy as np
import wave
from gmm import *
from sreutil import *
from mywave import *

# def WaveRead(file):
# 	'''
# 	Brief:
# 	read a .wav file and return its wave data
# 	'''
# 	f = wave.open(file,'rb')
# 	params = f.getparams()
# 	nchannels, sampwidth, framerate, nframes = params[:4]

# 	# the wave file format
# 	assert(nchannels==1)
# 	assert(sampwidth==2)
# 	assert(framerate==16000)

# 	str_data = f.readframes(nframes)
# 	f.close()

# 	wave_data = np.fromstring(str_data, dtype = np.short)

# 	return wave_data

# def WaveWrite(file, wave_data, nchannels, sampwidth, framerate, nframes):
# 	'''
# 	Brief:
# 	write wave data into a .wav file
# 	'''
# 	f = wave.open(file,'wb')
# 	f.setnchannels(nchannels)
# 	f.setsampwidth(sampwidth)
# 	f.setframerate(framerate)
# 	f.setnframes(nframes)
# 	f.writeframes(wave_data)
# 	f.close()


if __name__ == '__main__':
	print __doc__
	
	wav = mywave()

	# os.mkdir('train_data')
	# os.mkdir('test_data')
	dataDirs = os.listdir('data')
	for dataDir in dataDirs:
		print dataDir
		waveData = wav.WaveRead('data/'+dataDir)
		waveVadIdx = vad(waveData ** 2)
		waveData = waveData[waveVadIdx]
		trainWave = waveData[:int(0.8*waveData.shape[0])]
		testWave = waveData[-int(0.2*waveData.shape[0]):]
		wav.WaveWrite(r'train_data/'+r'train_'+dataDir,trainWave,1,2,16000,trainWave.shape[0])
		wav.WaveWrite(r'test_data/'+r'test_'+dataDir,testWave,1,2,16000,testWave.shape[0])

	ubm_data = 'train_data_for_UBM'
	os.mkdir(ubm_data)
	train_data = 'train_data'
	train_data_dir = os.listdir(train_data)
	for train_wave in train_data_dir:
		print train_wave
		if train_wave[-5] == 'M':
			waveData = wav.WaveRead(train_data+r'/'+train_wave)
			waveData = waveData[:int(0.2*waveData.shape[0])]
			wav.WaveWrite(ubm_data+r'/'+train_wave,waveData,1,2,16000,waveData.shape[0])
		elif train_wave[-5] == 'F':
			waveData = wav.WaveRead(train_data+r'/'+train_wave)
			wav.WaveWrite(ubm_data+r'/'+train_wave,waveData,1,2,16000,waveData.shape[0])
		
