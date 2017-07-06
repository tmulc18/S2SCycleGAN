import stft
import numpy as np

global SMP_RATE, SEG_SIZE
SMP_RATE = 16000

window_sec = .02 #window length 20 ms
SEG_SIZE = int(SMP_RATE*window_sec)

OVER_LAP = 2


def compute_spectrogram(wav_data):
	specgram = stft.spectrogram(wav_data,framelength=SEG_SIZE,overlap=OVER_LAP)
	reals,ims = np.real(specgram),np.imag(specgram)
	return reals

def compute_inverse_spectrogram(reals,ims=None):
	if ims != None:
		specgram = reals+1j*ims
	else:
		specgram = reals
	output = stft.ispectrogram(specgram,framelength=SEG_SIZE,overlap=OVER_LAP)
	return output

def get_seg_size():
	return SEG_SIZE
def get_overlap():
	return OVER_LAP