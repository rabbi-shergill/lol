import tensorflow as tf
import network
import numpy as np

tf.enable_eager_execution()

# input_tensor has dimensions [time_step, batch_size, vector_dimensions]
'''
Current Design
3 LSTM layers: 128, 128, 64
2 fully connected: 32(tanh), 10(softmax)
'''
def get_random_walk(side, min_steps, max_steps):		# side of grid, [min_steps, max_steps)
	x, y = np.random.randint(0, side, 2)
	visited = np.asarray([False for _ in range(side ** 2)]).reshape([side, side])
	steps = 


def get_sequence(batch):
	for item in batch:


EPOCHS = 100
LR = [0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5]
WINDOW_SIDE = 3
BACTH_SIZE = 8
### get the model

model = network.network(WINDOW_SIDE ** 2 + 2, [64, 128, 128], [64])

### load dataset MNIST
data = tf.keras.datasets.mnist.load_data()
(train_x, train_y), (test_x, test_y) = data

#try out pixel by pixel
for epoch in range(EPOCHS):

