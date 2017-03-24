import argparse
import cPickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)

# Export model parameters to h5 file
def save_h5(args, net, data_loader):
    # Begin tf session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))
        else:
            print 'checkpoint name not specified... exiting.'
            return

        vs = tf.get_collection(tf.GraphKeys.VARIABLES)
        vals = sess.run(vs)
        exclude = ["Adam", "beta", "learning_rate", "kl_weight"]

        if args.oracle:
            if args.recurrent:
                filename = 'oracle_lstm.h5'
            else:
                filename = 'oracle_mlp.h5'
        else:
            if args.recurrent:
                filename = 'bc_lstm.h5'
            else:
                filename = 'bc_fc.h5'
        
        with h5py.File(filename, 'a') as f:
            dset = f.create_group('policy')
            dset['obs_mean'] = data_loader.shift_s
            dset['obs_std'] = data_loader.scale_s
            dset['act_mean'] = data_loader.shift_a
            dset['act_std'] = data_loader.scale_a
            for v, val in safezip(vs, vals):
                if all([e not in v.name for e in exclude]):
                    print v.name
                    dset[v.name] = val



