import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cPickle
from dataloader import DataLoader
import h5py
import numpy as np
import os
import tensorflow as tf
import time
from utils import latent_viz, save_h5
import vae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./models', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')

    parser.add_argument('--batch_size',         type=int,   default= 64,        help='minibatch size')
    parser.add_argument('--state_dim',          type=int,   default=  51,       help='number of state variables')
    parser.add_argument('--action_dim',         type=int,   default=  2,        help='number of action variables')
    parser.add_argument('--z_dim',              type=int,   default=  2,        help='dimensions of latent variable')
    parser.add_argument('--sample_size',        type=int,   default=  10,       help='number of samples from z')

    parser.add_argument('--num_epochs',         type=int,   default= 50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.004,     help='learning rate')
    parser.add_argument('--decay_rate',         type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')
    parser.add_argument('--save_h5',            type=bool,  default=False,      help='Whether to save network params to h5 file')

    ###############################
    #          Encoder            #
    ###############################
    parser.add_argument('--encoder_size',          type=int,   default=128,        help='number of neurons in each LSTM layer')
    parser.add_argument('--num_encoder_layers',    type=int,   default=  2,        help='number of layers in the LSTM')
    parser.add_argument('--seq_length',            type=int,   default=100,        help='LSTM sequence length')

    ############################
    #       Policy Network     #
    ############################
    parser.add_argument('--policy_size',        type=int,   default=128,        help='number of neurons in each feedforward layer')
    parser.add_argument('--num_policy_layers',  type=int,   default=  2,        help='number of layers in the policy network')
    parser.add_argument('--input_dropout',      type=float, default=  1.0,      help='percent of inputs to keep')

    ############################
    #       Reconstructor      #
    ############################
    parser.add_argument('--rec_size',        type=int,   default= 64,        help='number of neurons in each feedforward layer')
    parser.add_argument('--num_rec_layers',  type=int,   default=  2,        help='number of layers in the policy network')
    parser.add_argument('--rec_weight',      type=float, default=  0.5,      help='weight applied to reconstruction cost')

    args = parser.parse_args()

    # Construct model
    net = vae.VariationalAutoencoder(args)

    # Export model parameters or perform training
    if args.save_h5:
        save_h5(args, net)
    else:
        train(args, net)

# Train network
def train(args, net):
    data_loader = DataLoader(args.batch_size, args.val_frac, args.seq_length)

    # Begin tf session
    with tf.Session() as sess:
        #Function to evaluate loss on validation set
        def val_loss():
            data_loader.reset_batchptr_val()
            policy_loss = 0.0
            for b in xrange(data_loader.n_batches_val):
                # Get batch of inputs/targets
                batch_dict = data_loader.next_batch_val()
                s = batch_dict["states"]
                a = batch_dict["actions"]
                _, _, state = net.encode(sess, s, a, args)

                # Set state and action input for encoder
                s_enc, a_enc = s[:,args.seq_length-1], a[:,args.seq_length-1]

                # Now loop over all timesteps, finding loss
                for t in xrange(args.seq_length):
                    # Get input and target values for specific time step (repeat values for multiple samples)
                    s_t, a_t = s[:,t], a[:,t]
                    s_t_rep = np.reshape(np.repeat(s_t, args.sample_size, axis=0), [args.batch_size, args.sample_size, args.state_dim])
                    a_t_rep = np.reshape(np.repeat(a_t, args.sample_size, axis=0), [args.batch_size, args.sample_size, args.action_dim])

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.states_encode] = s_enc
                    feed_in[net.actions_encode] = a_enc
                    feed_in[net.states] = s_t_rep
                    feed_in[net.actions] = a_t_rep
                    feed_in[net.kl_weight] = 0.01
                    for i, (c, m) in enumerate(net.lstm_state):
                        feed_in[c], feed_in[m] = state[i]
                    feed_out = net.policy_cost
                    out = sess.run(feed_out, feed_in)
                    policy_loss += out

            # Create new map of latent space
            return policy_loss/data_loader.n_batches_val

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Initialize variable to track validation score over time
        old_score = 1e6
        count_decay = 0
        decay_epochs = []

        # Initialize loss
        loss = 0.0
        policy_loss = 0.0
        rec_loss = 0.0

        # Set initial learning rate and weight on kl divergence
        print 'setting learning rate to ', args.learning_rate
        sess.run(tf.assign(net.learning_rate, args.learning_rate))
        kl_weight = 1e-4

        # Set up tensorboard summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('summaries/')

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))
            
            # Create plot of latent space
            latent_viz(args, net, e, sess, data_loader)

            # Set learning rate
            if (old_score - score) < 0.1 and kl_weight >= 0.005:
                count_decay += 1
                decay_epochs.append(e)
                if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                print 'setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay)
                sess.run(tf.assign(net.learning_rate, args.learning_rate * (args.decay_rate ** count_decay)))
            old_score = score

            data_loader.reset_batchptr_train()

            # Loop over batches
            for b in xrange(data_loader.n_batches_train):
                start = time.time()

                # Get batch of inputs/targets
                batch_dict = data_loader.next_batch_train()
                s = batch_dict["states"]
                a = batch_dict["actions"]
                _, _, state = net.encode(sess, s, a, args)

                # Set state and action input for encoder
                s_enc, a_enc = s[:,args.seq_length-1], a[:,args.seq_length-1,:args.action_dim]

                # Now loop over all timesteps, finding loss
                for t in xrange(args.seq_length):
                    # Get input and target values for specific time step (repeat values for multiple samples)
                    s_t, a_t = s[:,t], a[:,t]
                    s_t_rep = np.reshape(np.repeat(s_t, args.sample_size, axis=0), [args.batch_size, args.sample_size, args.state_dim])
                    a_t_rep = np.reshape(np.repeat(a_t, args.sample_size, axis=0), [args.batch_size, args.sample_size, args.action_dim])

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.states_encode] = s_enc
                    feed_in[net.actions_encode] = a_enc
                    feed_in[net.states] = s_t_rep
                    feed_in[net.actions] = a_t_rep
                    feed_in[net.kl_weight] = kl_weight
                    for i, (c, m) in enumerate(net.lstm_state):
                        feed_in[c], feed_in[m] = state[i]
                    feed_out = [net.cost, net.policy_cost, net.rec_cost, net.summary_policy, net.summary_encoder, net.summary_loss, net.train]
                    train_loss, policy_cost, rec_cost, summary_policy, summary_encoder, summary_loss, _ = sess.run(feed_out, feed_in)
                    
                    writer.add_summary(summary_policy, e * data_loader.n_batches_train + b)
                    writer.add_summary(summary_encoder, e * data_loader.n_batches_train + b)
                    writer.add_summary(summary_loss, e * data_loader.n_batches_train + b)
                    policy_loss += policy_cost
                    rec_loss += rec_cost
                    loss += train_loss

                end = time.time()

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, loss/10., end - start)
                    print "{}/{} (epoch {}), policy_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, policy_loss/10., end - start)
                    print "{}/{} (epoch {}), rec_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, rec_loss/10., end - start)
                    loss = 0.0
                    policy_loss = 0.0
                    rec_loss = 0.0
                kl_weight = min(0.01, kl_weight*1.005**(args.seq_length/300.))

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, 'vae.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
    main()
