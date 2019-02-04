# this is how to read the numpy array

import matplotlib.pyplot as plt
import numpy as np

d = np.load('npy_middle/0_0.npy')
print(d.shape)
plt.imshow((d[0:-2,0])[0]), plt.show()

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)
