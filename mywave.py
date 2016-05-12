import sys,os
import numpy as np
import wave

class mywave:
	"""
	Used to read and write wav files
	"""	
	def __init__(self):
		pass
	def WaveRead(self,file):
		'''
		Brief:
		read a .wav file and return its wave data
		'''
		f = wave.open(file,'rb')
		params = f.getparams()
		nchannels, sampwidth, framerate, nframes = params[:4]

		# the wave file format
		assert(nchannels==1)
		assert(sampwidth==2)
		assert(framerate==16000)

		str_data = f.readframes(nframes)
		f.close()

		wave_data = np.fromstring(str_data, dtype = np.short)

		return wave_data

	def WaveWrite(self,file, wave_data, nchannels, sampwidth, framerate, nframes):
		'''
		Brief:
		write wave data into a .wav file
		'''
		f = wave.open(file,'wb')
		f.setnchannels(nchannels)
		f.setsampwidth(sampwidth)
		f.setframerate(framerate)
		f.setnframes(nframes)
		f.writeframes(wave_data)
		f.close()



