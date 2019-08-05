import tensorflow as tf

def f():
	return tf.get_variable('name', shape = [2, 2])

with tf.variable_scope('test_scope'):
	print(f())