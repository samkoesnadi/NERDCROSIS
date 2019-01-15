# this is how to read the numpy array

import matplotlib.pyplot as plt
import numpy as np

d = np.load('npy_middle/0_0.npy')
plt.imshow(d[0][0]), plt.show()
