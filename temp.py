import numpy as np

a = np.arange(2 * 2 * 3).reshape([2, 2, 3])
b = np.reshape(b, [4, 3])
b = np.transpose(b, [1, 0])
print(b[0])