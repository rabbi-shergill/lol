import tensorflow as tf
import numpy as np

class fc:

	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		w_shape = [input_dims, output_dims]
		b_shape = [output_dims]
		
		self.w = tf.get_variable(name = 'w', shape = w_shape)
		self.b = tf.get_variable(name = 'b', shape = b_shape)

	def create_flow(self, input_tensor, activation = 'TANH'):
		lin_act = tf.matmul(input_tensor, self.w) + self.b
		if(activation == 'RELU'):
			return tf.nn.relu(lin_act)
		else if(activation == 'SIGMOID'):
			return tf.nn.sigmoid(lin_act)
		else if(activation == 'TANH'):
			return tf.nn.tanh(lin_act)
		else if(activation == 'SOFTMAX'):
			return tf.nn.softmax(lin_act)
		else
			return lin_act
	