import tensorflow as tf
import network
import numpy as np
from util import publish
from util import progress
from util import get_gather_map
import config
import numpy as np
import tensorflow.contrib.eager as tfe
#need to move to static graphs: this is not working
# input_tensor has dimensions [time_step, batch_size, vector_dimensions]
'''
Current Design
3 LSTM layers: 128, 128, 64
2 fully connected: 32(tanh), 10(softmax)
'''

# SETUP
tf.enable_eager_execution()

model = network.network(config.window_size() ** 2 + 2, config.lstm_layers(), config.fc_layers())
(images, labels), (t_images, t_labels) = tf.keras.datasets.mnist.load_data()
images = np.asarray(images[: 2000]).astype(np.float32) * 2.0 / 255.0 - 1.0
t_images = np.asarray(images[: 200]).astype(np.float32) * 2.0 / 255.0 - 1.0
labels = np.asarray(labels[: 2000])
t_labels = np.asarray(t_labels[: 200])
gather_map = tf.constant(get_gather_map(config.batch_size(), 28, config.window_size()))
global_step = 0



def train(epoch, global_step):
	optimizer = config.optimizer(steps = global_step, epoch = epoch)
	# return {'accuracy': 10.0, 'loss': 10.0}
	for batch in range(len(labels) // config.batch_size()):
		L = batch * config.batch_size()
		progress(epoch, (L * 10.0) / len(labels))
		R = L + config.batch_size()
		mini_batch_images = images[L: R]
		mini_batch_labels = labels[L: R]
		with tfe.GradientTape() as tape:
			s = tf.concat([tf.gather_nd(mini_batch_images, gather_map), tf.zeros(list(np.shape(gather_map)[: 2]) + [2])], -1)
			logits = model.flow(s)
			input_labels_one_hot = tf.one_hot(mini_batch_labels, 10)
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = input_labels_one_hot, logits = logits))

		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
		global_step += 1

	loss = 0.0
	success = 0.0
	for batch in range(len(labels) // config.batch_size()):
		L = batch * config.batch_size()
		R = L + config.batch_size()
		mini_batch_images = images[L: R]
		mini_batch_labels = labels[L: R]
		s = tf.concat([tf.gather_nd(mini_batch_images, gather_map), tf.zeros(list(np.shape(gather_map)[: 2]) + [2])], -1)
		logits = model.flow(s)
		classes = tf.argmax(logits, 1)
		loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = input_labels_one_hot, logits = logits))
		success += tf.reduce_sum(tf.cast(tf.equal(classes, mini_batch_labels), tf.float32))

	return {'accuracy': success * 100.0 / len(labels), 'loss': loss}

def test():
	success = 0.0
	for batch in range(len(t_labels) // config.batch_size()):
		L = batch * config.batch_size()
		R = L + config.batch_size()
		mini_batch_images = t_images[L: R]
		mini_batch_labels = t_labels[L: R]
		s = tf.concat([tf.gather_nd(mini_batch_images, gather_map), tf.zeros(list(np.shape(gather_map)[: 2]) + [2])], -1)
		logits = model.flow(s)
		classes = tf.argmax(logits, 1)
		success += tf.reduce_sum(tf.cast(tf.equal(classes, mini_batch_labels), tf.float32))

	return {'accuracy': success * 100.0 / len(t_labels)}


for epoch in range(config.epochs()):
	results = train(epoch, global_step)
	train_accuracy = results['accuracy']
	progress(0, 10.0)
	loss = results['loss']
	test_accuracy = test()['accuracy']
	publish(float(train_accuracy), float(loss), float(test_accuracy))