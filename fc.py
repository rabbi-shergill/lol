import tensorflow as tf
import numpy as np

class fc:

	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		w_shape = [input_dims, output_dims]
		b_shape = [output_dims]
		
		self.w = tf.get_variable(name = 'w', shape = w_shape, initializer = tf.glorot_normal_initializer())
		self.b = tf.get_variable(name = 'b', shape = b_shape, initializer = tf.glorot_normal_initializer())
		self.trainable_variables = [self.w, self.b]

	def flow(self, input_tensor, activation = 'TANH'):
		# print(np.shape(input_tensor[0]))
		lin_act = tf.matmul(input_tensor, self.w) + self.b
		if(activation == 'RELU'):
			return tf.nn.relu(lin_act)
		elif(activation == 'SIGMOID'):
			return tf.nn.sigmoid(lin_act)
		elif(activation == 'TANH'):
			return tf.nn.tanh(lin_act)
		elif(activation == 'SOFTMAX'):
			return tf.nn.softmax(lin_act)
		else:
			return lin_act
