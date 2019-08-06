import tensorflow as tf
import lstm
import fc

class network:

	def __init__(self, input_dims, lstm_dims, fc_dims):		# fc_dims is excluding the final 10 sized output layer
		last_dims = input_dims
		self.lstm_layers = []
		self.fc_layers = []
		self.trainable_variables = []
		count = 0
		for dims in input_dims:
			with tf.variable_scope('lstm_layer_' + str(count)):
				self.lstm_layers.append(lstm.lstm(last_dims, dims))
				self.trainable_variables += self.lstm_layers[-1]
			last_dims = dims
			count += 1
		for dims in fc_dims:
			with tf.variable_scope('fc_layer_' + str(count)):
				self.fc_layers.append(fc.fc(last_dims, dims))
				self.trainable_variables += self.fc_layers[-1]
			last_dims = dims
			count += 1

		self.output = fc.fc(last_dims, 10)

	def flow(self, input_tensors):
		output_tensor = input_tensors
		for lstm_layer in self.lstm_layers:
			output_tensor = lstm_layer.flow(output_tensor)
		for fc_layer in self.fc_layers[ -1]:
			output_tensor = fc_layer.flow(output_tensor)
		output_tensor = self.output.flow(output_tensor, 'SOFTMAX')
		return output_tensor
