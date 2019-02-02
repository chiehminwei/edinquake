from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import csv 

from edinquake import EdinquakeModel
from tensorflow.contrib.layers.python.layers import initializers
from collections import defaultdict

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, acoustic_signals, labels):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.acoustic_signals = acoustic_signals
    self.labels = labels

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids, token_start_mask):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.token_start_mask = token_start_mask


class DataProcessor(object):
  """Base class for data converters for sequence regression data sets."""

  def get_train_examples(self, data_dir):
	"""Gets a collection of `InputExample`s for the train set."""
	raise NotImplementedError()

  def get_dev_examples(self, data_dir):
	"""Gets a collection of `InputExample`s for the dev set."""
	raise NotImplementedError()

  def get_test_examples(self, data_dir):
	"""Gets a collection of `InputExample`s for the dev set."""
	raise NotImplementedError()

  def get_labels(self):
	"""Gets the list of labels for this data set."""
	raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
	"""Reads a tab separated value file."""
	with tf.gfile.Open(input_file, "r") as f:
	  reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
	  lines = []
	  for line in reader:
		  lines.append(line)
	  return lines

class EdinquakeProcessor(DataProcessor):
  """Processor for the CNCCG data set."""

  def get_train_examples(self, data_dir):
  	lines = self._read_tsv(os.path.join(data_dir,'train.csv'))
  	examples = []
  	for (i, line) in enumerate(lines):
  	  acoustic_signals, labels = line.split('\t')
  	  acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
  	  labels = [float(label) for label in labels.split(',')]

  	  guid = "train-%d" % (i)
  	  examples.append(
  		  InputExample(guid, acoustic_signals, labels))
  	return examples

  def get_dev_examples(self, data_dir):
	"""See base class."""
  	lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
  	examples = []
  	for (i, line) in enumerate(lines):
  	  acoustic_signals, labels = line.split('\t')
  	  acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
  	  labels = [float(label) for label in labels.split(',')]

  	  guid = "dev-%d" % (i)
  	  examples.append(
  		  InputExample(guid, acoustic_signals, labels))
  	return examples

  def get_test_examples(self, data_dir):
	"""See base class."""
  	lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
  	true_labels = []
  	examples = []
  	for (i, line) in enumerate(lines):
  	  acoustic_signals, labels = line.split('\t')
  	  acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
  	  labels = [float(label) for label in labels.split(',')]

  	  guid = "test-%d" % (i)
  	  examples.append(
  		  InputExample(guid, acoustic_signals, labels))
  	return examples

def convert_single_example():

def file_based_convert_examples_to_features():

def file_based_input_fn_builder():

def create_model():
	model = EdinquakeModel()
	rst = model.compute()
	return rst

def model_fn_builder():

	def model_fn():

  return model_fn
