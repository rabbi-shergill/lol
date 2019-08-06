import tensorflow as tf

def batch_size():
	return 8

def window_size():
	return 2

def epochs():
	return 100

def learning_rate(loss, epoch, steps):
	return 0.01

def optimizer(loss = 0, epoch = 0, steps = 0):		#epoch and steps sent in this case
	return tf.train.AdamOptimizer(learning_rate(loss, epoch, steps))

def lstm_layers():
	return [128, 128, 64]

def fc_layers():
	return [32]