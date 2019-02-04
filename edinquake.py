# encoding=utf-8

"""
edinquake basic rnn model
@Author:s1882834
"""

import tensorflow as tf
from tensorflow.contrib import rnn


class EdinquakeModel(object):
		def __init__(self, input_batch, output_dim,
								 rnn_hidden_size, rnn_num_layers, rnn_dropout_rate, 
								 mlp_hidden_size, mlp_num_layers, mlp_dropout_rate,
								 initializers, labels, length_mask, l2_scale, is_training):
				if not is_training:
						rnn_dropout_rate = 0.0
						mlp_dropout_rate = 0.0

				if not l2_scale:
						l2_scale = 0.1

				self.input_batch = input_batch
				self.output_dim = 1
				self.rnn_hidden_size = rnn_hidden_size
				self.rnn_num_layers = rnn_num_layers
				self.rnn_dropout_rate = rnn_dropout_rate
				self.mlp_hidden_size = mlp_hidden_size
				self.mlp_num_layers = mlp_num_layers
				self.mlp_dropout_rate = mlp_dropout_rate
				self.initializers = initializers
				self.labels = labels
				self.length_mask = length_mask
				self.l2_scale = l2_scale
				self.is_training = is_training


		def compute(self):
				rnn_output = self._rnn_layer(self.input_batch)
				loss, predictions = self._mlp_layer(rnn_output)
				return loss, predictions

		def _rnn_cell(self):
				cell = rnn.LayerNormBasicLSTMCell(self.rnn_hidden_size, dropout_keep_prob=1-self.rnn_dropout_rate)
				return cell

		def _rnn_layer(self, input_batch):
				with tf.variable_scope('rnn_layer'):
						cell = self._rnn_cell()
						if self.rnn_num_layers > 1:
								cell = rnn.MultiRNNCell([cell] * self.rnn_num_layers, state_is_tuple=True)
							 
						outputs, _ = tf.nn.dynamic_rnn(cell, input_batch, dtype=tf.float32)
				return outputs

		def _mlp_layer(self, prev_output):
				with tf.variable_scope('mlp_layer'):
						regularizer = tf.nn.l2_regularizer(scale=self.l2_scale)
						for layer_idx in range(self.mlp_num_layers):
								with tf.variable_scope("MLP_layer_%d" % layer_idx):
										layer_input = prev_output
										layer_output =  tf.layers.dense(
											layer_input,
											self.mlp_hidden_size,
											tf.nn.relu,
											kernel_initializer=self.initializers.xavier_initializer(),
											kernel_regularizer=regularizer)
										layer_output = tf.layers.batch_normalization(layer_output, training=self.is_training)
										layer_output = dropout(layer_output, self.mlp_dropout_rate)
										prev_output = layer_output
						# map to scalar
						predictions = tf.layers.dense(prev_output, self.output_dim, tf.nn.relu, 
														kernel_initializer=self.initializers.xavier_initializer(),
														kernel_regularizer=regularizer)
						loss = tf.losses.mean_squared_error(self.labels, predictions, self.length_mask)
						l2_loss = tf.losses.get_regularization_loss()
						loss += l2_loss

						return loss, predictions

def dropout(input_tensor, dropout_prob):
	"""Perform dropout.

	Args:
		input_tensor: float Tensor.
		dropout_prob: Python float. The probability of dropping out a value (NOT of
			*keeping* a dimension as in `tf.nn.dropout`).

	Returns:
		A version of `input_tensor` with dropout applied.
	"""
	if dropout_prob is None or dropout_prob == 0.0:
		return input_tensor

	output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output