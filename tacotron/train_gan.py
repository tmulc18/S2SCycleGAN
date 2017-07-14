# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os

import librosa
from tqdm import tqdm

from data_load import get_batch, get_batch_eval
from hyperparams import Hyperparams as hp
from modules import *
from networks import encode_dis, encode, decode1, decode2
import numpy as np
from prepro import *
import tensorflow as tf
from utils import shift_by_one, restore_shape, spectrogram2wav
from tensorflow.python.client import timeline
from scipy.io.wavfile import write


class Graph:
    # Load vocabulary 
    #char2idx, idx2char = load_vocab()
    
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.q,self.y, self.z, self.num_batch = get_batch()
            else: # Evaluation
                self.x = tf.placeholder(tf.float32, shape=(None, None,hp.n_mels*hp.r))
                self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))

            #self.decoder_inputs = shift_by_one(self.y) 
            
            with tf.variable_scope("Generator"):
                # Encoder
                self.memory_gen = encode(self.q, is_training=is_training) # (N, T, E)
                
                # Decoder 
                decode_length = int((hp.bin_size_y[1]*hp.sr-(hp.win_length-1))/((hp.hop_length)*hp.r)) # about 50
                self._outputs1_gen = tf.zeros([hp.batch_size,1,hp.n_mels*hp.r])
                outputs1_gen_list = []
                for j in range(decode_length):
                    if j == 0:
                        reuse = None
                    else: 
                        reuse = True
                    self._outputs1_gen += decode1(self._outputs1_gen,
                                            self.memory_gen,
                                            is_training=is_training,reuse=reuse)
                    outputs1_gen_list.append(self._outputs1_gen)
                self.outputs1_gen = tf.concat(outputs1_gen_list,1)
                self.outputs2_gen = decode2(self.outputs1_gen,is_training=is_training)


            with tf.variable_scope("Discriminator"):
                self.final_state_real = encode_dis(self.z, is_training=is_training)
                self.final_state_fake = encode_dis(self.outputs2_gen, is_training=is_training,reuse=True)



            if is_training:
                # Discriminator Loss
                self.dis_loss_real = tf.reduce_mean(tf.squared_difference(self.final_state_real,1))
                self.dis_loss_fake = tf.reduce_mean(tf.squared_difference(self.final_state_fake,0))
                self.dis_loss = tf.reduce_mean(self.dis_loss_real + self.dis_loss_fake)

                # Generator Loss
                self.gen_loss = tf.reduce_mean(tf.squared_difference(self.final_state_fake,1))

                                                
                # Training Scheme
                dvars = [e for e in self.graph.get_collection('trainable_variables') if 'Discriminator' in e.name]
                gvars = [e for e in self.graph.get_collection('trainable_variables') if 'Generator' in e.name]

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

                grad_d,var_d = zip(*self.optimizer.compute_gradients(self.dis_loss,var_list=dvars))
                grad_d_clipped ,_= tf.clip_by_global_norm(grad_d,5.)
                grad_g,var_g = zip(*self.optimizer.compute_gradients(self.gen_loss,var_list=gvars))
                grad_g_clipped ,_= tf.clip_by_global_norm(grad_g,5.)
                self.train_op_dis=self.optimizer.apply_gradients(zip(grad_d_clipped,var_d))
                self.train_op_gen=self.optimizer.apply_gradients(zip(grad_g_clipped,var_g))
                # self.train_op_dis = self.optimizer.minimize(self.dis_loss, global_step=self.global_step,var_list=dvars)
                # self.train_op_gen = self.optimizer.minimize(self.gen_loss, global_step=self.global_step,var_list=gvars)

                # Profiling
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # Summmary 
                tf.summary.scalar('dis_loss_real', self.dis_loss_real)
                tf.summary.scalar('dis_loss_fake', self.dis_loss_fake)
                tf.summary.scalar('dis_loss', self.dis_loss)
                tf.summary.scalar('gen_loss', self.gen_loss)
                
                
                self.merged = tf.summary.merge_all()
         
def sample_audio(g,sess):
    """
    Samples audio from the generator from training examples

    Parameters:

    g : TensorFlow Graph

    sess : TensorFlow Session
    """
    mname = 'gan'
    og,act,gen = sess.run([g.q,g.z,g.outputs2_gen])
    for i,(s0,s1,s2) in enumerate(zip(og,act,gen)):
        s0 = restore_shape(s0, hp.win_length//hp.hop_length, hp.r)
        s1 = restore_shape(s1, hp.win_length//hp.hop_length, hp.r)
        s2 = restore_shape(s2, hp.win_length//hp.hop_length, hp.r)           
        # generate wav files
        if hp.use_log_magnitude:
            audio0 = spectrogram2wav(np.power(np.e, s0)**hp.power)
            audio1 = spectrogram2wav(np.power(np.e, s1)**hp.power)
            audio2 = spectrogram2wav(np.power(np.e, s2)**hp.power)
        else:
            s0 = np.where(s0 < 0, 0, s0)
            s1 = np.where(s1 < 0, 0, s1)
            s2 = np.where(s2 < 0, 0, s2)
            audio0 = spectrogram2wav(s0**hp.power)
            audio1 = spectrogram2wav(s1**hp.power)
            audio2 = spectrogram2wav(s2**hp.power)
        write(hp.outputdir + "/gan_{}_org.wav".format(i), hp.sr, audio0)
        write(hp.outputdir + "/gan_{}_act.wav".format(i), hp.sr, audio1)
        write(hp.outputdir + "/gan_{}_gen.wav".format(i), hp.sr, audio2)

def main():   
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        gpu_options = tf.GPUOptions(allow_growth=True)
        run_metadata = tf.RunMetadata()
        config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
        with sv.managed_session(config=config) as sess:
            print('made it to training')
            for epoch in tqdm(range(1, hp.num_epochs+1)): 
                if sv.should_stop(): 
                    print("Something is broken");break

                # Sampling Audio
                if epoch % hp.audio_summary == 1:
                    print("Sampling")
                    sample_audio(g,sess)

                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    for _ in range(hp.k):
                        sess.run(g.train_op_dis,options=options,run_metadata=run_metadata)
                    sess.run(g.train_op_gen,options=options,run_metadata=run_metadata)

                    # #Profile Logging
                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # with open('timeline/timeline_01_step_%d.json' % step, 'w') as f:
                    #     f.write(chrome_trace)

                
                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

if __name__ == '__main__':
    main()
    print("Done")
