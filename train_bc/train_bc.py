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
from utils import save_h5
import bc_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./models', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')

    parser.add_argument('--batch_size',         type=int,   default= 64,        help='minibatch size')
    parser.add_argument('--state_dim',          type=int,   default= 51,        help='number of state variables')
    parser.add_argument('--num_classes',        type=int,   default= 4,         help='number of driver classes')
    parser.add_argument('--action_dim',         type=int,   default= 2,         help='number of action variables')

    parser.add_argument('--num_epochs',         type=int,   default= 50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.004,     help='learning rate')
    parser.add_argument('--decay_rate',         type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')
    parser.add_argument('--save_h5',            type=bool,  default= False,     help='Whether to save network params to h5 file')

    parser.add_argument('--seq_length',         type=int,   default=50,         help='Sequence length for training')
    parser.add_argument('--burn_in_length',     type=int,   default=10,         help='Amount of time steps for initializing LSTM internal state')

    ############################
    #       Policy Network     #
    ############################
    parser.add_argument('--policy_size',        type=int,   default= 128,       help='number of neurons in each layer')
    parser.add_argument('--num_policy_layers',  type=int,   default= 2,         help='number of layers in the policy network')
    parser.add_argument('--recurrent',          type=bool,  default= False,     help='whether to use recurrent policy')
    parser.add_argument('--oracle',             type=bool,  default= False,     help='whether to include class as input to policy')
    parser.add_argument('--dropout_level',      type=float, default=  1.0,      help='dropout applied to fc policy')

    args = parser.parse_args()

    # Construct model
    net = bc_policy.BCPolicy(args)

    # Export model parameters or perform training
    if args.save_h5:
        data_loader = DataLoader(args.batch_size, args.val_frac, args.seq_length + args.burn_in_length, args.oracle)
        save_h5(args, net, data_loader)
    else:
        train(args, net)

# Train network
def train(args, net):
    data_loader = DataLoader(args.batch_size, args.val_frac, args.seq_length + args.burn_in_length, args.oracle)

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

                # Get state/action pairs for initialize internal state of lstm
                if args.recurrent:
                    s_init, a_init = s[:,:args.burn_in_length], a[:,:args.burn_in_length]
                    state = net.burn_in(sess, s_init, a_init, args)

                # Now loop over all timesteps, finding loss
                for t in xrange(args.burn_in_length, args.seq_length):
                    # Get input and target values for specific time step (repeat values for multiple samples)
                    s_t, a_t = s[:,t], a[:,t]

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.states] = s_t
                    feed_in[net.actions] = a_t

                    if args.recurrent:
                        for i, (c, m) in enumerate(net.lstm_state):
                            feed_in[c], feed_in[m] = state[i]

                        feed_out = [net.cost]
                        for c, m in net.final_state:
                            feed_out.append(c)
                            feed_out.append(m)
                    else:
                        feed_out = net.cost

                    # Make forward pass
                    out = sess.run(feed_out, feed_in)
                    if args.recurrent:
                        cost = out[0]
                        state_flat = out[1:]
                        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
                    else:
                        cost  = out
                    
                    # Increment cost
                    policy_loss += cost

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

        # Set initial learning rate
        print 'setting learning rate to ', args.learning_rate
        sess.run(tf.assign(net.learning_rate, args.learning_rate))

        # Set up tensorboard summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('summaries/')

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if (old_score - score) < 0.01:
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

                # Get state/action pairs for initialize internal state of lstm
                if args.recurrent:
                    s_init, a_init = s[:,:args.burn_in_length], a[:,:args.burn_in_length]
                    state = net.burn_in(sess, s_init, a_init, args)

                # Now loop over all timesteps, finding loss
                for t in xrange(args.burn_in_length, args.seq_length):
                    # Get input and target values for specific time step (repeat values for multiple samples)
                    s_t, a_t = s[:,t], a[:,t]

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.states] = s_t
                    feed_in[net.actions] = a_t

                    if args.recurrent:
                        for i, (c, m) in enumerate(net.lstm_state):
                            feed_in[c], feed_in[m] = state[i]
                    feed_out = [net.cost, net.train]
                    if args.recurrent:
                        for c, m in net.final_state:
                            feed_out.append(c)
                            feed_out.append(m)

                    out = sess.run(feed_out, feed_in)
                    cost  = out[0]
                    if args.recurrent:
                        state_flat = out[2:]
                        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
                        
                    loss += cost

                end = time.time()

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, loss/10., end - start)
                    loss = 0.0

            # Save model every epoch
            if args.oracle:
                if args.recurrent:
                    checkpoint_path = os.path.join(args.save_dir, 'oracle_recurrent.ckpt')
                else:
                    checkpoint_path = os.path.join(args.save_dir, 'oracle_mlp.ckpt')
            else:
                if args.recurrent:
                    checkpoint_path = os.path.join(args.save_dir, 'bc_recurrent.ckpt')
                else:
                    checkpoint_path = os.path.join(args.save_dir, 'bc_fc.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
    main()
