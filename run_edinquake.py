from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import csv 
import optimization

from edinquake import EdinquakeModel
from tensorflow.contrib.layers.python.layers import initializers
from collections import defaultdict

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, acoustic_signals, labels, length_mask):
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
		self.length_mask = length_mask

# class InputFeatures(object):
#   """A single set of features of data."""
#   def __init__(self, inputs, labels):
#     self.inputs = inputs
#     self.labels = labels

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
			reader = csv.reader(f, delimiter=",", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

# TO DO: PROCESS THE DATA ACCORDING TO MAX_SEQ_LENGTH
class EdinquakeProcessor(DataProcessor):
	"""Processor for the edinquake data set."""
	def __init__(self, max_seq_length):
		self.max_seq_length = max_seq_length

	def get_train_examples(self, data_dir):
		lines = self._read_tsv(os.path.join(data_dir, "quakes_128_real"))
		examples = []
		for (i, line) in enumerate(lines):
			acoustic_signals, labels = line.split('\t')
			acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
			labels = [float(label) for label in labels.split(',')]

			guid = "train-%d" % (i)
			examples.append(
				InputExample(guid, acoustic_signals, labels, length_mask))
		return examples

	# def get_train_examples(self, data_dir):
	# 	lines = self._read_csv(os.path.join(data_dir,'train.csv'))
	# 	examples = []

	# 	acoustic_signals = []
	# 	labels = []
	# 	length_mask = []
	# 	guid_counter = 0
	# 	for (i, line) in enumerate(lines):
	# 		acoustic_signal, label = [float(string) for string in line.split(',')]
	# 		acoustic_signals.append(acoustic_signal)
	# 		labels.append(label)
	# 		length_mask.append(1.0)
	# 		if len(acoustic_signals) == self.max_seq_length:
	# 			guid = "train-%d" % (guid_counter)
	# 			examples.append(
	# 				InputExample(guid, acoustic_signals, labels, length_mask))
	# 			guid_counter += 1
	# 			acoustic_signals = []
	# 			labels = []
	# 			length_mask = []

	# 	assert len(acoustic_signals) <= self.max_seq_length
	# 	if len(acoustic_signals) < self.max_seq_length:
	# 		diff = self.max_seq_length - len(acoustic_signals)
	# 		acoustic_signals += [0.0] * diff
	# 		labels += [0.0] * diff
	# 		length_mask += [0.0] * diff
	# 		guid = "train-%d" % (guid_counter)
	# 		examples.append(
	# 				InputExample(guid, acoustic_signals, labels, length_mask))

	# 	return examples

	def get_dev_examples(self, data_dir, max_seq_length):
		"""See base class."""
		lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
		examples = []
		for (i, line) in enumerate(lines):
			acoustic_signals, labels = line.split('\t')
			acoustic_signals = [float(signal) for signal in acoustic_signals.split(',')]
			labels = [float(label) for label in labels.split(',')]

			guid = "dev-%d" % (i)
			examples.append(
				InputExample(guid, acoustic_signals, labels, length_mask))
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
				InputExample(guid, acoustic_signals, labels, length_mask))
		return examples

# def convert_single_example():

def file_based_convert_examples_to_features(examples, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""
	writer = tf.python_io.TFRecordWriter(output_file)

	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		# feature = convert_single_example(ex_index, example)
		feature = example

		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f
		features = collections.OrderedDict()
		features["inputs"] = create_int_feature(feature.acoustic_signals)
		features["labels"] = create_int_feature(feature.labels)
		features["length_mask"] = create_int_feature(feature.length_mask)
		
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	name_to_features = {
			"inputs": tf.FixedLenFeature([seq_length], tf.int64),
			"labels": tf.FixedLenFeature([seq_length], tf.int64),
			"length_mask": tf.FixedLenFeature([seq_length], tf.int64)
	}

	def _decode_record(record, name_to_features):
		"""Decodes a record to a TensorFlow example."""
		example = tf.parse_single_example(record, name_to_features)
		# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
		# So cast all int64 to int32.
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t

		return example

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]

		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)

		d = d.apply(
				tf.contrib.data.map_and_batch(
						lambda record: _decode_record(record, name_to_features),
						batch_size=batch_size,
						drop_remainder=drop_remainder))

		return d

	return input_fn

def create_model(is_training, inputs, labels, rnn_hidden_size,
											rnn_num_layers, rnn_dropout, mlp_hidden_size, mlp_num_layers, mlp_dropout, l2_scale, length_mask):
	model = EdinquakeModel(inputs, 1,
								 rnn_hidden_size, rnn_num_layers, rnn_dropout, 
								 mlp_hidden_size, mlp_num_layers, mlp_dropout,
								 initializers, labels, length_mask, l2_scale, is_training)
	rst = model.compute()
	return rst

def model_fn_builder(learning_rate, num_train_steps, num_warmup_steps, use_tpu, rnn_hidden_size,
											rnn_num_layers, rnn_dropout, mlp_hidden_size, mlp_num_layers, mlp_dropout, l2_scale):

	def model_fn(features, labels, mode, params):
		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
		inputs = features["inputs"]
		labels = features["labels"]
		length_mask = features["length_mask"]

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		(total_loss, predictions) = create_model(is_training, inputs, labels, 
																rnn_hidden_size, rnn_num_layers, rnn_dropout, 
																mlp_hidden_size, mlp_num_layers, mlp_dropout, 
																l2_scale, length_mask)

		def tpu_scaffold():
				return tf.train.Scaffold()

		scaffold_fn = tpu_scaffoldm
		output_spec = None
		if mode == tf.estimator.ModeKeys.TRAIN:

			train_op = optimization.create_optimizer(
					total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
					mode=mode,
					loss=total_loss,
					train_op=train_op,
					scaffold_fn=scaffold_fn)

		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(labels, predictions):
				mse = tf.metrics.mean_squared_error(labels, predictions)
				return {
						"mea_squared_error": mse,
				}

			eval_metrics = (metric_fn, [labels, predictions])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
					mode=mode,
					loss=total_loss,
					eval_metrics=eval_metrics,
					scaffold_fn=scaffold_fn)
		else:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				 mode=mode,
				 predictions={"predictions": predictions},
				 scaffold_fn=scaffold_fn)
		return output_spec

	return model_fn
