#/usr/bin/python2
# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

import codecs
import csv
import os
import re
import glob
import scipy.io.wavfile as wavfile

from hyperparams import Hyperparams as hp
import numpy as np


# def load_vocab():
#     vocab = "EG abcdefghijklmnopqrstuvwxyz'" # E: Empty. ignore G
#     char2idx = {char:idx for idx, char in enumerate(vocab)}
#     idx2char = {idx:char for idx, char in enumerate(vocab)}
#     return char2idx, idx2char    

def getWaveName(wavepath):
    return wavepath.split('/')[-1]

def findWave(wavefile,path):
    r = glob.glob(path+wavefile)
    return r[0]

def create_train_data():
    m2f = dict()
    for i in glob.glob(hp.mpath+'/*.wav'):
        m2f[i]=findWave(getWaveName(i),hp.fpath)

    bin_mapping = {}
    for filename in glob.glob(hp.mpath+'*.wav'):
        fm, wav_data = wavfile.read(filename)
        size = wav_data.shape[0]/(1.0*fm)
        if size < hp.bin_size_x[1] and size > hp.bin_size_x[0]:
            fm2, wav_data2 = wavfile.read(m2f[filename])
            size2 = wav_data2.shape[0]/(1.0*fm2)
            if size2 < hp.bin_size_y[1] and size2 > hp.bin_size_y[0]:
                bin_mapping[filename] = m2f[filename]

    return bin_mapping.keys(),bin_mapping.values()
    # Load vocabulary
    # char2idx, idx2char = load_vocab()  
    # texts, sound_files = [], []
    # reader = csv.reader(codecs.open(hp.text_file, 'rb', 'utf-8'))
    # for row in reader:
    #     sound_fname, text, duration = row
    #     sound_file = hp.sound_fpath + "/" + sound_fname + ".wav"
    #     text = re.sub(r"[^ a-z']", "", text.strip().lower())
         
    #     if hp.min_len <= len(text) <= hp.max_len:
    #         texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
    #         sound_files.append(sound_file)
             
    # return texts, sound_files
     
def load_train_data():
    """We train on the whole data but the last num_samples."""
    texts, sound_files = create_train_data()
    if hp.sanity_check: # We use a single mini-batch for training to overfit it.
        texts, sound_files = texts[:hp.batch_size]*1, sound_files[:hp.batch_size]*1
    else:
        texts, sound_files = texts[:-hp.num_samples], sound_files[:-hp.num_samples]
    return texts, sound_files
 
def load_eval_data():
    """We evaluate on the last num_samples."""
    texts, _ = create_train_data()
    if hp.sanity_check: # We generate samples for the same texts as the ones we've used for training.
        texts = texts[:hp.batch_size]
    else:
        texts = texts[-hp.num_samples:]
    
    X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32) # byte to int 
        X[i, :len(_text)] = _text
    
    return X
 

