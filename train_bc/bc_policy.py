import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.contrib.layers.python.layers import initializers

class BCPolicy():
    def __init__(self, args):
        # Placeholder for data
        if args.oracle:
            self.states = tf.placeholder(tf.float32, [args.batch_size, args.state_dim + args.num_classes], name="states_encode")
        else:
            self.states = tf.placeholder(tf.float32, [args.batch_size, args.state_dim], name="states_encode")
        self.actions = tf.placeholder(tf.float32, [args.batch_size, args.action_dim], name="actions_encode")
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Create the computational graph
        if args.recurrent:
            self._create_lstm_policy(args)
        else:
            self._create_fc_policy(args)
        self._create_optimizer(args)

    def _create_lstm_policy(self, args):
        # Create LSTM portion of network
        lstm = rnn_cell.LSTMCell(args.policy_size, state_is_tuple=True, initializer=initializers.xavier_initializer())
        self.full_lstm = rnn_cell.MultiRNNCell([lstm] * args.num_policy_layers, state_is_tuple=True)
        self.lstm_state = self.full_lstm.zero_state(args.batch_size, tf.float32)

        # Forward pass
        policy_input = self.states
        output, self.final_state = seq2seq.rnn_decoder([policy_input], self.lstm_state, self.full_lstm)
        output = tf.reshape(tf.concat(1, output), [-1, args.policy_size])

        # Fully connected layer to latent variable distribution parameters
        W = tf.get_variable("lstm_w", [args.policy_size, args.action_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("lstm_b", [args.action_dim])
        self.a_mean = tf.nn.xw_plus_b(output, W, b)

        # Initialize logstd
        self.a_logstd = tf.Variable(np.zeros(args.action_dim), name="a_logstd", dtype=tf.float32)

    def _create_fc_policy(self, args):

        # Construct policy input
        policy_input = self.states

        # Create fully connected network of desired size
        if args.oracle:
            W = tf.get_variable("w_0", [args.state_dim + args.num_classes, args.policy_size], initializer=initializers.xavier_initializer())
        else:
            W = tf.get_variable("w_0", [args.state_dim, args.policy_size], initializer=initializers.xavier_initializer())
        b = tf.get_variable("b_0", [args.policy_size])
        output = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(policy_input, W, b)), args.dropout_level)

        for i in xrange(1, args.num_policy_layers):
            W = tf.get_variable("w_"+str(i), [args.policy_size, args.policy_size], initializer=initializers.xavier_initializer())
            b = tf.get_variable("b_"+str(i), [args.policy_size])
            output = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(output, W, b)), args.dropout_level)

        W = tf.get_variable("w_end", [args.policy_size, args.action_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("b_end", [args.action_dim])
        self.a_mean = tf.nn.xw_plus_b(output, W, b)

        # Initialize logstd
        self.a_logstd = tf.Variable(np.zeros(args.action_dim), name="a_logstd", dtype=tf.float32)

    def _create_optimizer(self, args):
        # Find negagtive log-likelihood of true actions
        std_a = tf.exp(self.a_logstd) + 1e-3
        pl_1 = 0.5 * tf.to_float(args.action_dim) * np.log(2. * np.pi)
        pl_2 = tf.reduce_sum(tf.log(std_a))
        pl_3 = 0.5 * tf.reduce_sum(tf.square((self.actions - self.a_mean)/std_a), 1)
        policy_loss = tf.reduce_mean(pl_1 + pl_2 + pl_3)

        # Find overall loss
        self.cost = tf.reduce_mean(policy_loss/args.seq_length)
        self.summary_loss = tf.summary.scalar("Overall loss", self.cost)

        # Perform parameter update
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.apply_gradients(zip(grads, tvars))

    def burn_in(self, sess, s, a, args):
        # Initialize the internal state
        state = []
        for c, m in self.lstm_state:
            state.append((c.eval(session= sess), m.eval(session= sess)))

        # Loop over burn-in time to initilize internal state of lstm
        for t in xrange(args.burn_in_length):
            # Get state and action values for specific time step
            s_t, a_t = s[:,t], a[:,t]

            # Construct inputs to network
            feed_in = {}
            feed_in[self.states] = s_t
            feed_in[self.actions] = a_t
            for i, (c, m) in enumerate(self.lstm_state):
                feed_in[c], feed_in[m] = state[i]

            # Define outputs
            feed_out = [self.a_mean]
            for c, m in self.final_state:
                feed_out.append(c)
                feed_out.append(m)

            # Make pass
            res = sess.run(feed_out, feed_in)
            state_flat = res[1:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

        return state





