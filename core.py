# input_tensor has dimensions [time_step, batch_size, vector_dimensions]
'''
Current Design
3 LSTM layers: 128, 128, 64
2 fully connected: 32(tanh), 10(softmax)
'''

import tensorflow as tf
import network
from argparse import ArgumentParser
import numpy as np



def train(model_directory):
	tf.enable_eager_execution()
	

if __name__ == '__main__':
	argparse = ArgumentParser()
	argparse.add_argument('--model_directory', type = str, help = 'Path to the directory where the checkpoint files are stored', default = './checkpoints')
	argparse.add_argument('--train', type = str, help = 'true/false - to train or not', default = 'false')
	args = argparse.parse_args()
	if(args.train == 'true'):
		train(args.model_directory)