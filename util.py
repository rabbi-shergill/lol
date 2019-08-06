import numpy as np

def get_gather_map(batch_size, image_size, window_size):
	from itertools import product
	gather_map = []
	for batch in range(batch_size):
		gm = []
		for x in range(0, image_size, window_size):
			for y in range(0, image_size, window_size):
				temp = np.asarray(list(product(range(x, x + window_size), range(y, y + window_size))))
				gm.append(np.concatenate((np.full([len(temp), 1], batch), temp), axis = 1))
		gather_map.append(gm)
	return np.transpose(np.asarray(gather_map), (1, 0, 2, 3))

def publish(train_accuracy, loss, test_accuracy):
	print(' --- Train-Accuracy = {:10.3f} --- Loss = {:10.3f} --- Test-Accuracy = {:10.3f}'.format(train_accuracy, loss, test_accuracy))

def progress(epoch, perc):
	fill = int(perc)
	bar = ''.join(['>' for _ in range(fill)])
	bar += ''.join(['-' for _ in range(10 - len(bar))])
	print('\rEpoch = {:3} --- [{}] --- {:5.2f}%'.format(epoch, bar, perc * 10.0), end = '')