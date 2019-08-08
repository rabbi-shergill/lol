import tensorflow as tf

def batch_size():
	return 64

def window_size():
	return 2

def epochs():
	return 100

def learning_rate(epoch, steps):	#
# 	if(epoch > 5):
# 		return 0.01
	return 0.01

def lstm_layers():
	return [128, 128, 64]

def fc_layers():
	return [32]
