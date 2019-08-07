import tensorflow as tf
import network
import numpy as np
from util import publish
from util import progress
from util import gather
import config
import numpy as np

# input_tensor has dimensions [time_step, batch_size, vector_dimensions]
# Current Design
# 3 LSTM layers: 128, 128, 64
# 2 fully connected: 32(tanh), 10(softmax)
# SETUP

print('Loading and preparing data', end = '')
(images, labels), (t_images, t_labels) = tf.keras.datasets.mnist.load_data()
images = np.asarray(images[: 5000]).astype(np.float32) * 2.0 / 255.0 - 1.0
t_images = np.asarray(t_images[: 1000]).astype(np.float32) * 2.0 / 255.0 - 1.0
labels = np.asarray(labels[: 5000])
t_labels = np.asarray(t_labels[: 1000])
images = gather(images, config.window_size())
t_images = gather(t_images, config.window_size())
TIME_STEPS = (28 // config.window_size()) ** 2
images = np.concatenate((images, np.zeros(list(np.shape(images)[: -1]) + [2], np.float32)), axis = -1)
t_images = np.concatenate((t_images, np.zeros(list(np.shape(t_images)[: -1]) + [2], np.float32)), axis = -1)
print('\rData Loaded' + '                           ')
# print(np.shape(images), np.shape(t_images))
# exit(0)

print('Preparing Model', end = '')
input_images = tf.placeholder(tf.float32, shape = [TIME_STEPS, None, config.window_size() ** 2 + 2])
input_tensor_list = [input_images[i] for i in range(TIME_STEPS)]
input_labels = tf.placeholder(tf.int32, shape = [None])

model = network.network(config.window_size() ** 2 + 2, config.lstm_layers(), config.fc_layers())
# gather_map = tf.constant(get_gather_map(config.batch_size(), 28, config.window_size()))
global_step = 0

# timestamped_decomposed_images = tf.concat(tf.gather_nd(input_images, gather_map), tf.zeros(list(gather_map.get_shape()[: 2]) + [2]), -1)
learning_rate = tf.placeholder(tf.float32)
logits = model.flow(input_tensor_list, config.batch_size())
input_labels_one_hot = tf.one_hot(input_labels, 10)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = input_labels_one_hot, logits = logits))
classes = tf.cast(tf.argmax(logits, 1), tf.int32)
success = tf.reduce_sum(tf.cast(tf.equal(classes, input_labels), tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


session = tf.Session()
session.run(tf.global_variables_initializer())
print('\rModel Initialised')


def train(epoch, global_step):
	print('Epoch {:3d} {:3.2f}'.format(epoch, 0), end = '')
	for batch in range(len(labels) // config.batch_size()):
		L = batch * config.batch_size()
		R = L + config.batch_size()
		mini_batch_images = images[: ,L: R, :]
		mini_batch_labels = labels[L: R]
		feed_dict = {input_images: mini_batch_images, input_labels: mini_batch_labels, learning_rate: config.learning_rate(epoch = epoch, steps = global_step)}
		session.run(optimizer, feed_dict = feed_dict)
		# grads = tape.gradient(loss, model.trainable_variables)
		# optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
		global_step += 1
		print('\rEpoch {:3d} {:3.2f}'.format(epoch, L * 100.0 / len(labels)), end = '')
	print('\rEpoch {:3d} {:3.2f}'.format(epoch, 100.0), end = '')

def test(images, labels):
	correct_predictions = 0.0
	total_predictions = 0.0
	model_loss = 0.0
	print('\rEpoch {:3d} {:3.2f}'.format(epoch, 0.0), end = '')
	for batch in range(len(labels) // config.batch_size()):
		L = batch * config.batch_size()
		R = L + config.batch_size()
		mini_batch_images = images[:, L: R, :]
		mini_batch_labels = labels[L: R]
		# s = tf.concat([tf.gather_nd(mini_batch_images, gather_map), tf.zeros(list(np.shape(gather_map)[: 2]) + [2])], -1)
		# logits = model.flow(s)
		feed_dict = {input_images: mini_batch_images, input_labels: mini_batch_labels}
		output = session.run([loss, success], feed_dict = feed_dict)
		total_predictions += config.batch_size()
		correct_predictions += output[1]
		model_loss += output[0]
		print('\rEpoch {:3d} {:3.2f}'.format(epoch, L * 100.0 / len(labels)), end = '')
	print('\rEpoch {:3d} {:3.2f}'.format(epoch, 100.0), end = '')
	return {'accuracy': correct_predictions * 100.0 / total_predictions, 'loss': model_loss}


for epoch in range(config.epochs()):
	train(epoch, global_step)
	train_data_results = test(images, labels)
	test_data_results = test(t_images, t_labels)
	print('\rEpoch {} {} {}'.format(epoch, train_data_results, test_data_results))
