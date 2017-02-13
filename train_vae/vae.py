import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.contrib.layers.python.layers import initializers

class VariationalAutoencoder():
    def __init__(self, args):
        # Placeholder for data
        self.states_encode = tf.placeholder(tf.float32, [args.batch_size, args.state_dim], name="states_encode")
        self.actions_encode = tf.placeholder(tf.float32, [args.batch_size, args.action_dim], name="actions_encode")
        self.states = tf.placeholder(tf.float32, [args.batch_size, args.sample_size, args.state_dim], name="states")
        self.actions = tf.placeholder(tf.float32, [args.batch_size, args.sample_size, args.action_dim], name="actions")
        self.rec_in = tf.placeholder(tf.float32, [args.batch_size, args.sample_size, args.action_dim], name="actions")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Create the computational graph
        self._create_encoder(args)
        self._create_policy(args)
        self._create_reconstructor(args)
        self._create_optimizer(args)

    def _create_encoder(self, args):
        # Create LSTM portion of network
        lstm = rnn_cell.LSTMCell(args.encoder_size, state_is_tuple=True, initializer=initializers.xavier_initializer())
        self.full_lstm = rnn_cell.MultiRNNCell([lstm] * args.num_encoder_layers, state_is_tuple=True)
        self.lstm_state = self.full_lstm.zero_state(args.batch_size, tf.float32)

        # Forward pass
        encoder_input = tf.concat(1, [self.states_encode, self.actions_encode])
        output, self.final_state = seq2seq.rnn_decoder([encoder_input], self.lstm_state, self.full_lstm)
        output = tf.reshape(tf.concat(1, output), [-1, args.encoder_size])

        # Fully connected layer to latent variable distribution parameters
        W = tf.get_variable("latent_w", [args.encoder_size, 2*args.z_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("latent_b", [2*args.z_dim])
        logits = tf.nn.xw_plus_b(output, W, b)

        # Separate into mean and logstd
        self.z_mean, self.z_logstd = tf.split(1, 2, logits)

    def _create_policy(self, args):
        # Get samples from standard normal distribution, transform to match z-distribution
        samples = tf.random_normal([args.sample_size, args.batch_size, args.z_dim], name="z_samples")
        self.z_samples = samples * tf.exp(self.z_logstd) + self.z_mean
        self.z_samples = tf.transpose(self.z_samples, perm=[1, 0, 2])

        # Construct encoder input
        self.states_do = tf.nn.dropout(self.states, args.input_dropout)
        enc_in = tf.concat(2, [self.states_do, self.z_samples])
        enc_in = tf.reshape(enc_in, [args.batch_size*args.sample_size, args.state_dim + args.z_dim], name="enc_in")

        # Create fully connected network of desired size
        W = tf.get_variable("w_0", [args.state_dim + args.z_dim, args.policy_size], initializer=initializers.xavier_initializer())
        b = tf.get_variable("b_0", [args.policy_size])
        output = tf.nn.relu(tf.nn.xw_plus_b(enc_in, W, b))

        for i in xrange(1, args.num_policy_layers):
            W = tf.get_variable("w_"+str(i), [args.policy_size, args.policy_size], initializer=initializers.xavier_initializer())
            b = tf.get_variable("b_"+str(i), [args.policy_size])
            output = tf.nn.relu(tf.nn.xw_plus_b(output, W, b))

        W = tf.get_variable("w_end", [args.policy_size, args.action_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("b_end", [args.action_dim])
        a_mean = tf.nn.xw_plus_b(output, W, b)
        self.a_mean = tf.reshape(a_mean, [args.batch_size, args.sample_size, args.action_dim], name="a_mean")

        # Initialize logstd
        self.a_logstd = tf.Variable(np.zeros(args.action_dim), name="a_logstd", dtype=tf.float32)

    def _create_reconstructor(self, args):
        # Get samples from standard normal distribution, transform to match a-distribution
        samples = tf.random_normal([args.batch_size, args.sample_size, args.action_dim], name="a_samples")
        self.a_samples = samples * tf.exp(self.a_logstd) + self.a_mean
        # self.a_samples = tf.transpose(self.a_samples, perm=[1, 0, 2])

        # Construct reconstructor input
        rec_in = tf.concat(2, [self.states, self.a_samples])
        rec_in = tf.reshape(rec_in, [args.batch_size*args.sample_size, args.state_dim + args.action_dim], name="rec_in")

        # Create fully connected network of desired size
        W = tf.get_variable("rec_w_0", [args.state_dim + args.action_dim, args.rec_size], initializer=initializers.xavier_initializer())
        b = tf.get_variable("rec_b_0", [args.rec_size])
        output = tf.nn.relu(tf.nn.xw_plus_b(rec_in, W, b))

        for i in xrange(1, args.num_rec_layers):
            W = tf.get_variable("rec_w_"+str(i), [args.rec_size, args.rec_size], initializer=initializers.xavier_initializer())
            b = tf.get_variable("rec_b_"+str(i), [args.rec_size])
            output = tf.nn.relu(tf.nn.xw_plus_b(output, W, b))

        W = tf.get_variable("rec_w_end", [args.rec_size, args.z_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("rec_b_end", [args.z_dim])
        z_rec_mean = tf.nn.xw_plus_b(output, W, b)
        self.z_rec_mean = tf.reshape(z_rec_mean, [args.batch_size, args.sample_size, args.action_dim], name="z_rec_mean")

        # Initialize logstd
        self.z_rec_logstd = tf.Variable(np.zeros(args.z_dim), name="z_rec_logstd", dtype=tf.float32)

    def _create_optimizer(self, args):
        # Find negagtive log-likelihood of true actions
        std_a = tf.exp(self.a_logstd) + 1e-3
        pl_1 = 0.5 * tf.to_float(args.action_dim) * np.log(2. * np.pi)
        pl_2 = tf.reduce_sum(tf.log(std_a))
        pl_3 = 0.5 * tf.reduce_sum(tf.square((self.actions - self.a_mean)/std_a), 2)
        policy_loss = tf.reduce_mean(pl_1 + pl_2 + pl_3, 1)

        # Find KL-divergence between prior (standard normal) and approximate posterior
        std_z = tf.exp(self.z_logstd) + 1e-3
        el_1 = -0.5 * tf.to_float(args.z_dim)
        el_2 = -tf.reduce_sum(tf.log(std_z), 1)
        el_3 = tf.reduce_sum(tf.square(std_z), 1)
        el_4 = tf.reduce_sum(tf.square(self.z_mean), 1)
        encoder_loss = el_1 + el_2 + el_3 + el_4

        # Find negagtive log-likelihood of reconstructed z-values
        std_z_rec = tf.exp(self.z_rec_logstd) + 1e-3
        rl_1 = 0.5 * tf.to_float(args.z_dim) * np.log(2. * np.pi)
        rl_2 = tf.reduce_sum(tf.log(std_z_rec))
        rl_3 = 0.5 * tf.reduce_sum(tf.square((self.z_samples - self.z_rec_mean)/std_z_rec), 2)
        rec_loss = tf.reduce_mean(rl_1 + rl_2 + rl_3, 1)

        # Find overall loss
        self.policy_cost = tf.reduce_mean(policy_loss/args.seq_length)
        self.rec_cost = tf.reduce_mean(rec_loss/args.seq_length)
        self.cost = tf.reduce_mean((policy_loss + self.kl_weight*encoder_loss + args.rec_weight*rec_loss)/args.seq_length)
        self.summary_policy = tf.summary.scalar("Policy loss", tf.reduce_mean(policy_loss)/args.seq_length)
        self.summary_encoder = tf.summary.scalar("Encoder loss", tf.reduce_mean(encoder_loss)/args.seq_length)
        self.summary_loss = tf.summary.scalar("Overall loss", self.cost)

        # Perform parameter update
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.apply_gradients(zip(grads, tvars))

    def encode(self, sess, s, a, args):
        # Initialize the internal state
        state = []
        for c, m in self.lstm_state:
            state.append((c.eval(session= sess), m.eval(session= sess)))

        # Loop over all timesteps, find posterior over z
        for t in xrange(args.seq_length-1):
            # Get state and action values for specific time step
            s_enc, a_enc = s[:,t], a[:,t]

            # Construct inputs to network
            feed_in = {}
            feed_in[self.states_encode] = s_enc
            feed_in[self.actions_encode] = a_enc
            for i, (c, m) in enumerate(self.lstm_state):
                feed_in[c], feed_in[m] = state[i]

            # Define outputs
            feed_out = [self.z_mean, self.z_logstd]
            for c, m in self.final_state:
                feed_out.append(c)
                feed_out.append(m)

            # Make pass
            res = sess.run(feed_out, feed_in)
            z_mean = res[0]
            z_logstd = res[1]
            state_flat = res[2:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

        return z_mean, z_logstd, state





