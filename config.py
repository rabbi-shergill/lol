import tensorflow as tf

def batch_size():
	return 8

def window_size():
	return 2

def epochs():
	return 100

def learning_rate(epoch, steps):
	if(epoch > 40):
		return 0.001
	return 0.1

def lstm_layers():
	return [128, 128]

def fc_layers():
	return []