# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing Language Model text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

class CharReader(object):

  def __init__(self, data_path, val_to_train_ratio=5):

    self.raw_data = open(data_path, 'r').read()

    self.val_to_train_ratio = val_to_train_ratio

    #Build Vocabulary and Dictionaries
    counter = collections.Counter(self.raw_data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.unique_tokens, _ = list(zip(*count_pairs))
    self.token_to_id = dict(zip(self.unique_tokens, range(len(self.unique_tokens))))
    self.vocab_size = len(self.unique_tokens)

    #print len(self.unique_tokens)

    #Convert Raw tokens to digits
    self.data_as_ids = []
    for id, token in enumerate(self.raw_data):
      if token in self.token_to_id.keys():
        self.data_as_ids.append(self.token_to_id[token])
      else:
        self.data_as_ids.append(0)


    #Create Data Indexes
    data_size = len(self.data_as_ids)
    training_data_ratio = 100 - (self.val_to_train_ratio*2)
    self.train_data_max_index = (data_size * training_data_ratio)//100
    self.valid_data_min_index = self.train_data_max_index+1
    self.valid_data_max_index = ((data_size * self.val_to_train_ratio)//100)+self.train_data_max_index
    self.test_data_min_index = self.valid_data_max_index+1


  def print_data_info(self):
    print('----------------------------------------')
    print('Data total tokens: %d tokens' % (len(self.raw_data)))
    print('Data vocabulary size: %d tokens' % (len(self.unique_tokens)))
    print('Training Data total tokens: %d tokens' % (len(self.get_training_data())))
    print('Validation Data total tokens: %d tokens' % (len(self.get_validation_data())))
    print('Test Data total tokens: %d tokens' % (len(self.get_test_data())))
    print('----------------------------------------')

  def _read_words(self, filename):
    with gfile.GFile(filename, "r") as f:
      return f.read().replace("\n", " <eos> ").split()




  def convertDigitsToText(self, token_predicted_per_batch,  first_token=" "):

    for token_predicted_per_batch_item in token_predicted_per_batch:
      first_token.join(self.id_to_token[token_predicted_per_batch_item] )

    return first_token



  def get_training_data(self):
    return self.data_as_ids[0:self.train_data_max_index]

  def get_validation_data(self):
    return self.data_as_ids[self.valid_data_min_index:self.valid_data_max_index]

  def get_test_data(self):
    length = len(self.data_as_ids)-1
    return self.data_as_ids[self.test_data_min_index:length]

  def generateXYPairs(self, raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    batch_len = len(raw_data) // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    for i in range((batch_len - 1) // num_steps):
      x = data[:, i*num_steps:(i+1)*num_steps]
      y = data[:, i*num_steps+1:(i+1)*num_steps+1]
      yield (x, y)



  def limit_data_size(self, original_data, max_size=None):
    if max_size==None:
      return original_data

    print("Limiting Input data size to:  %d tokens" % (max_size))
    return original_data[0:max_size]


class WordReader(object):

  def __init__(self, data_path, val_to_train_ratio=5):

    self.raw_data = open(data_path, 'r').read()

    self.val_to_train_ratio = val_to_train_ratio

    #Build Vocabulary and Dictionaries
    words_data = self._read_words(data_path)
    words_data.append('')
    counter = collections.Counter(words_data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    self.unique_tokens, _ = list(zip(*count_pairs))
    self.token_to_id = dict(zip(self.unique_tokens, range(len(self.unique_tokens))))
    self.vocab_size = len(self.unique_tokens)

    #print len(self.unique_tokens)

    #Convert Raw tokens to digits
    self.data_as_ids = [self.token_to_id[word] for word in words_data if word in self.token_to_id]
    # for id, token in enumerate(self.raw_data):
    #   if token in self.token_to_id:
    #     self.data_as_ids.append(self.token_to_id[token])
    #   else:
    #     self.data_as_ids.append(0)


    #Create Data Indexes
    data_size = len(self.data_as_ids)
    training_data_ratio = 100 - (self.val_to_train_ratio*2)
    self.train_data_max_index = (data_size * training_data_ratio)//100
    self.valid_data_min_index = self.train_data_max_index+1
    self.valid_data_max_index = ((data_size * self.val_to_train_ratio)//100)+self.train_data_max_index
    self.test_data_min_index = self.valid_data_max_index+1


  def print_data_info(self):
    print('----------------------------------------')
    print('Data total tokens: %d tokens' % (len(self.raw_data)))
    print('Data vocabulary size: %d tokens' % (len(self.unique_tokens)))
    print('Training Data total tokens: %d tokens' % (len(self.get_training_data())))
    print('Validation Data total tokens: %d tokens' % (len(self.get_validation_data())))
    print('Test Data total tokens: %d tokens' % (len(self.get_test_data())))
    print('----------------------------------------')

  def _read_words(self, filename):
    with gfile.GFile(filename, "r") as f:
       #return f.read().decode("utf-8").replace("\n", "<eos>").split()
      return f.read().replace("\n", " <eos> ").split()


  def convertDigitsToText(self, token_predicted_per_batch,  first_token=" "):

    for token_predicted_per_batch_item in token_predicted_per_batch:
      first_token.join(self.id_to_token[token_predicted_per_batch_item] )

    return first_token


  def get_training_data(self):
    return self.data_as_ids[0:self.train_data_max_index]

  def get_validation_data(self):
    return self.data_as_ids[self.valid_data_min_index:self.valid_data_max_index]

  def get_test_data(self):
    length = len(self.data_as_ids)-1
    return self.data_as_ids[self.test_data_min_index:length]

  def generateXYPairs(self, raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    batch_len = len(raw_data) // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    for i in range((batch_len - 1) // num_steps):
      x = data[:, i*num_steps:(i+1)*num_steps]
      y = data[:, i*num_steps+1:(i+1)*num_steps+1]
      yield (x, y)



  def limit_data_size(self, original_data, max_size=None):
    if max_size==None:
      return original_data

    print("Limiting Input data size to:  %d tokens" % (max_size))
    return original_data[0:max_size]
