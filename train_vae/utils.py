import argparse
import cPickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time
import vae

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
        exclude = ["Adam", "beta", "learning_rate", "kl_weight", "rec"]
        encoder = ["rnn_decoder", "latent"]

        # Save encoder parameters
        with h5py.File('encoder_new.h5', 'a') as f:
            dset = f.create_group('encoder')
            dset['obs_mean'] = data_loader.shift_s
            dset['obs_std'] = data_loader.scale_s
            for v, val in safezip(vs, vals):
                if all([e not in v.name for e in exclude]):
                    if any([e in v.name for e in encoder]):
                        print v.name
                        dset[v.name] = val

        # Save policy parameters
        print 'policy: '
        with h5py.File('policy_vae_new.h5', 'a') as f:
            dset = f.create_group('policy')
            dset['obs_mean'] = data_loader.shift_s
            dset['obs_std'] = data_loader.scale_s
            dset['act_mean'] = data_loader.shift_a
            dset['act_std'] = data_loader.scale_a
            for v, val in safezip(vs, vals):
                if all([e not in v.name for e in exclude]):
                    if all([e not in v.name for e in encoder]):
                        print v.name
                        dset[v.name] = val

def gen_samples(args, net, sess, batchptr, n_batches, next_batch_dict, name):
    # Generate passive samples
    batchptr = 0
    full_sample = 0.0
    print 'generating z samples ' + name + '...'
    for b in xrange(n_batches/2):
        batch_dict = next_batch_dict[name]()
        s = batch_dict["states"]
        a = batch_dict["actions"]
        z_mean, z_logstd, _ = net.encode(sess, s, a, args)

        samples = np.random.normal(size=(args.sample_size, args.batch_size, args.z_dim))
        z_samples = samples * np.exp(z_logstd) + z_mean
        z_samples = np.reshape(z_samples, (args.sample_size*args.batch_size, args.z_dim))

        if type(full_sample) is float:
            full_sample = z_samples
        else:
            full_sample = np.concatenate((full_sample, z_samples), axis=0)

    return full_sample

# Visualize samples from latent space
def latent_viz(args, net, e, sess, data_loader):
    # Create dict of functions to choose next batch
    next_batch_dict = {'passive': data_loader.next_batch_pass, 'medium1': data_loader.next_batch_med1, 
                        'medium2': data_loader.next_batch_med2, 'aggressive': data_loader.next_batch_agg}

    # Generate samples 
    data_loader.batchptr_pass = 0
    data_loader.batchptr_med1 = 0
    data_loader.batchptr_med2 = 0
    data_loader.batchptr_agg = 0
    full_sample_pass = gen_samples(args, net, sess, data_loader.batchptr_pass, data_loader.n_batches_pass, next_batch_dict, 'passive')
    full_sample_med1 = gen_samples(args, net, sess, data_loader.batchptr_med1, data_loader.n_batches_med1, next_batch_dict, 'medium1')
    full_sample_med2 = gen_samples(args, net, sess, data_loader.batchptr_med2, data_loader.n_batches_med2, next_batch_dict, 'medium2')
    full_sample_agg = gen_samples(args, net, sess, data_loader.batchptr_agg, data_loader.n_batches_agg, next_batch_dict, 'aggressive')

    # Select random subset of values in full set of samples
    ind_pass = random.sample(xrange(len(full_sample_pass)), 2000)
    ind_agg = random.sample(xrange(len(full_sample_agg)), 2000)
    ind_med1 = random.sample(xrange(len(full_sample_med1)), 2000)
    ind_med2 = random.sample(xrange(len(full_sample_med2)), 2000)

    # Plot and save results
    print 'saving and exiting.'
    plt.cla()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(full_sample_pass[ind_pass, 0], full_sample_pass[ind_pass, 1], 'bx', label='Passive')
    plt.plot(full_sample_agg[ind_agg, 0], full_sample_agg[ind_agg, 1], 'ro', label='Aggressive')
    plt.plot(full_sample_med1[ind_med1, 0], full_sample_med1[ind_med1, 1], 'gp', label='Medium 1')
    plt.plot(full_sample_med2[ind_med2, 0], full_sample_med2[ind_med2, 1], 'cd', label='Medium 2')

    # plt.plot(full_sample_pass[ind_pass], 2.0*np.ones(2000), 'bx', label='Passive')
    # plt.plot(full_sample_agg[ind_agg], np.ones(2000), 'ro', label='Aggressive')
    # plt.plot(full_sample_med1[ind_med1], -1.0*np.ones(2000), 'gp', label='Medium 1')
    # plt.plot(full_sample_med2[ind_med2], -2.0*np.ones(2000), 'cd', label='Medium 2')

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.xlabel(r'$z_1$', fontsize=16)
    plt.ylabel(r'$z_2$', fontsize=16)
    plt.title('Epoch ' + str(e))
    plt.legend(loc='upper right', numpoints=1)
    plt.grid()
    plt.savefig('./images_recurrent/latent_viz_'+str(e)+'.png')

