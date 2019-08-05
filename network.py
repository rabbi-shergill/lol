import tensorflow as tf
import lstm
import fc

class encoder:

	def __init__(self, input_dims, lstm_dims, fc_dims):
		last_dims = input_dims
		self.lstm_layers = []
		self.fc_layers = []
		count = 0
		for dims in input_dims:
			with tf.variable_scope('lstm_layer_' + str(count)):
				self.lstm_layers.append(lstm.lstm(last_dims, dims))
			last_dims = dims
			count += 1
		for dims in fc_dims:
			with tf.variable_scope('fc_layer_' + str(count)):
				self.fc_layers.append(fc.fc(last_dims, dims))
			last_dims = dims
			count += 1

	def create_flow(self, input_tensor):