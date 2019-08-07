import tensorflow as tf
import numpy as np

class lstm:

	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		w_shape = [input_dims, output_dims]
		u_shape = [output_dims, output_dims]
		b_shape = [output_dims]
		
		# forget gate parameters
		self.wf = tf.get_variable(name = 'lstm_w_forget_gate', shape = w_shape)
		self.uf = tf.get_variable(name = 'lstm_u_forget_gate', shape = u_shape)
		self.bf = tf.get_variable(name = 'lstm_b_forget_gate', shape = b_shape)
		# input gate parameters
		self.wi = tf.get_variable(name = 'lstm_w_input_gate', shape = w_shape)
		self.ui = tf.get_variable(name = 'lstm_u_input_gate', shape = u_shape)
		self.bi = tf.get_variable(name = 'lstm_b_input_gate', shape = b_shape)
		# output gate parameters
		self.wo = tf.get_variable(name = 'lstm_w_output_gate', shape = w_shape)
		self.uo = tf.get_variable(name = 'lstm_u_output_gate', shape = u_shape)
		self.bo = tf.get_variable(name = 'lstm_b_output_gate', shape = b_shape)
		# cell state parameters
		self.wc = tf.get_variable(name = 'lstm_w_cell_state', shape = w_shape)
		self.uc = tf.get_variable(name = 'lstm_u_cell_state', shape = u_shape)
		self.bc = tf.get_variable(name = 'lstm_b_cell_state', shape = b_shape)

		self.trainable_variables = [self.wf, self.uf, self.bf, self.wi, self.ui, self.bi, self.wo, self.uo, self.bo, self.wc, self.uc, self.bc]

	def flow(self, input_tensors, batch_size = 1):
		h = tf.constant(0.0, shape = [batch_size, self.output_dims])
		c = tf.constant(0.0, shape = [batch_size, self.output_dims])
		H = []
		for input_tensor in input_tensors:
			ft = tf.nn.sigmoid(tf.matmul(input_tensor, self.wf) + tf.matmul(h, self.uf) + self.bf)
			it = tf.nn.sigmoid(tf.matmul(input_tensor, self.wi) + tf.matmul(h, self.ui) + self.bi)
			ot = tf.nn.sigmoid(tf.matmul(input_tensor, self.wo) + tf.matmul(h, self.uo) + self.bo)
			c = ft * c + it * tf.nn.tanh(tf.matmul(input_tensor, self.wc) + tf.matmul(h, self.uc) + self.bc)
			h = ot * tf.nn.tanh(c)
			H.append(h)
		return H
