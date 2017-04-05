import collections
import cPickle
import h5py
import math
import numpy as np
import os
import random

# Class to load and preprocess data
class DataLoader():
    def __init__(self, batch_size, val_frac, seq_length):
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seq_length = seq_length

        print 'validation fraction: ', self.val_frac

        print "loading data..."
        self._load_data()

        print 'creating splits...'
        self._create_split()

        print 'shifting/scaling data...'
        self._shift_scale()

    def _trim_data(self, full_s, full_a, intervals):
        # Python indexing; find bounds on data given seq_length
        intervals -= 1
        lengths = np.floor(np.diff(np.append(intervals, len(full_s)-1))/self.seq_length)*self.seq_length
        intervals = np.vstack((intervals, intervals + lengths)).T.astype(int)
        ret_bounds = np.insert(np.cumsum(lengths), 0, 0.).astype(int)

        # Remove states that don't fit due to value of seq_length
        s = np.zeros((int(sum(lengths)), full_s.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            s[ret_bounds[i]:ret_bounds[i+1]] = full_s[intervals[i, 0]:intervals[i, 1]]
        s = np.reshape(s, (-1, self.seq_length, full_s.shape[1]))

        # Remove actions that don't fit due to value of seq_length
        a = np.zeros((int(sum(lengths)), full_a.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            a[ret_bounds[i]:ret_bounds[i+1]] = full_a[intervals[i, 0]:intervals[i, 1]]
        a = np.reshape(a, (-1, self.seq_length, full_a.shape[1]))

        return s, a

    def _load_data(self):
        data_dir = '../2d_drive_data/'

        # Load mixed data
        filename = data_dir + 'data_distinct_drivers_4.jld'
        data = h5py.File(filename, 'r')
        s = data['features'][:]
        a = data['targets'][:]
        c = data['classes'][:]
        intervals = data['intervals'][:]
        data.close()

        # Augment state with class for trimming
        s = np.hstack((s, np.expand_dims(c, axis=1)))
        
        # Trim data based on intervals
        s, a = self._trim_data(s, a, intervals)
        
        # Make sure batch_size divides into num of examples 
        self.s = s[:int(np.floor(len(s)/self.batch_size)*self.batch_size)]
        self.s = np.reshape(self.s, (-1, self.batch_size, self.seq_length, s.shape[2]))
        self.a = a[:int(np.floor(len(a)/self.batch_size)*self.batch_size)]
        self.a = np.reshape(self.a, (-1, self.batch_size, self.seq_length, a.shape[2]))

        # Now separate states and classes
        self.c = self.s[:, :, :, 51]
        self.s = self.s[:, :, :, :51]
        
        # Print tensor shapes
        print 'states: ', self.s.shape
        print 'actions: ', self.a.shape
        print 'classes: ', self.c.shape

        # Create batch_dict
        self.batch_dict = {}
        self.batch_dict["states"] = np.zeros((self.batch_size, self.seq_length, s.shape[2]))
        self.batch_dict["actions"] = np.zeros((self.batch_size, self.seq_length, a.shape[2]))
        self.batch_dict["classes"] = np.zeros((self.batch_size, self.seq_length))

        # Shuffle data
        print 'shuffling...'
        p = np.random.permutation(len(self.s))
        self.s = self.s[p]
        self.a = self.a[p]
        self.c = self.c[p]


        # Load passive data
        filename = data_dir + 'data_passive.jld'
        data = h5py.File(filename, 'r')
        s_pass = data['features'][:]
        a_pass = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        s_pass, a_pass = self._trim_data(s_pass, a_pass, intervals)

        # Make sure batch_size divides into num of examples 
        self.s_pass = s_pass[:int(np.floor(len(s_pass)/self.batch_size)*self.batch_size)]
        self.s_pass = np.reshape(self.s_pass, (-1, self.batch_size, self.seq_length, s_pass.shape[2]))
        self.a_pass = a_pass[:int(np.floor(len(a_pass)/self.batch_size)*self.batch_size)]
        self.a_pass = np.reshape(self.a_pass, (-1, self.batch_size, self.seq_length, a_pass.shape[2]))


        # Load aggressive data
        filename = data_dir + 'data_aggressive.jld'
        data = h5py.File(filename, 'r')
        s_agg = data['features'][:]
        a_agg = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        s_agg, a_agg = self._trim_data(s_agg, a_agg, intervals)

        # Make sure batch_size divides into num of examples 
        self.s_agg = s_agg[:int(np.floor(len(s_agg)/self.batch_size)*self.batch_size)]
        self.s_agg = np.reshape(self.s_agg, (-1, self.batch_size, self.seq_length, s_agg.shape[2]))
        self.a_agg = a_agg[:int(np.floor(len(a_agg)/self.batch_size)*self.batch_size)]
        self.a_agg = np.reshape(self.a_agg, (-1, self.batch_size, self.seq_length, a_agg.shape[2]))

        # Load medium 1 data
        filename = data_dir + 'data_medium1.jld'
        data = h5py.File(filename, 'r')
        s_med1 = data['features'][:]
        a_med1 = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        s_med1, a_med1 = self._trim_data(s_med1, a_med1, intervals)

        # Make sure batch_size divides into num of examples 
        self.s_med1 = s_med1[:int(np.floor(len(s_med1)/self.batch_size)*self.batch_size)]
        self.s_med1 = np.reshape(self.s_med1, (-1, self.batch_size, self.seq_length, s_med1.shape[2]))
        self.a_med1 = a_med1[:int(np.floor(len(a_med1)/self.batch_size)*self.batch_size)]
        self.a_med1 = np.reshape(self.a_med1, (-1, self.batch_size, self.seq_length, a_med1.shape[2]))

        # Load medium 2 data
        filename = data_dir + 'data_medium2.jld'
        data = h5py.File(filename, 'r')
        s_med2 = data['features'][:]
        a_med2 = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        s_med2, a_med2 = self._trim_data(s_med2, a_med2, intervals)

        # Make sure batch_size divides into num of examples 
        self.s_med2 = s_med2[:int(np.floor(len(s_med2)/self.batch_size)*self.batch_size)]
        self.s_med2 = np.reshape(self.s_med2, (-1, self.batch_size, self.seq_length, s_med2.shape[2]))
        self.a_med2 = a_med2[:int(np.floor(len(a_med2)/self.batch_size)*self.batch_size)]
        self.a_med2 = np.reshape(self.a_med2, (-1, self.batch_size, self.seq_length, a_med2.shape[2]))

    # Separate data into train/validation sets
    def _create_split(self):

        # compute number of batches
        self.n_batches = len(self.s)
        self.n_batches_val = int(math.floor(self.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val
        self.n_batches_pass = len(self.s_pass)
        self.n_batches_agg = len(self.s_agg)
        self.n_batches_med1 = len(self.s_med1)
        self.n_batches_med2 = len(self.s_med2)

        print 'num training batches: ', self.n_batches_train
        print 'num validation batches: ', self.n_batches_val

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self):
        # Find means and std, ignore state values that are indicators
        self.shift_s = np.mean(self.s[:self.n_batches_train], axis=(0, 1, 2))
        self.scale_s = np.std(self.s[:self.n_batches_train], axis=(0, 1, 2))
        self.shift_a = np.mean(self.a[:self.n_batches_train], axis=(0, 1, 2))
        self.scale_a = np.std(self.a[:self.n_batches_train], axis=(0, 1, 2))

        # Get rid of scale for indicator features
        self.scale_s = np.array([1.0*(s < 1e-3) + s for s in self.scale_s])

        # Transform data
        self.s = (self.s - self.shift_s)/self.scale_s
        self.a = (self.a - self.shift_a)/self.scale_a

        self.s_pass = (self.s_pass - self.shift_s)/self.scale_s
        self.a_pass = (self.a_pass - self.shift_a)/self.scale_a

        self.s_agg = (self.s_agg - self.shift_s)/self.scale_s
        self.a_agg = (self.a_agg - self.shift_a)/self.scale_a

        self.s_med1 = (self.s_med1 - self.shift_s)/self.scale_s
        self.a_med1 = (self.a_med1 - self.shift_a)/self.scale_a

        self.s_med2 = (self.s_med2 - self.shift_s)/self.scale_s
        self.a_med2 = (self.a_med2 - self.shift_a)/self.scale_a

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train-1
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]
        self.batch_dict["classes"] = self.c[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

    # Sample a new batch of data from passive set
    def next_batch_pass(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_pass[self.batchptr_pass]
        self.batch_dict["actions"] = self.a_pass[self.batchptr_pass]

        # Update pointer
        self.batchptr_pass += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_agg(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_agg[self.batchptr_agg]
        self.batch_dict["actions"] = self.a_agg[self.batchptr_agg]

        # Update pointer
        self.batchptr_agg += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_med1(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_med1[self.batchptr_med1]
        self.batch_dict["actions"] = self.a_med1[self.batchptr_med1]

        # Update pointer
        self.batchptr_med1 += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_med2(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_med2[self.batchptr_med2]
        self.batch_dict["actions"] = self.a_med2[self.batchptr_med2]

        # Update pointer
        self.batchptr_med2 += 1
        return self.batch_dict

