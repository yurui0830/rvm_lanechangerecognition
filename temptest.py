import numpy as np

theta = np.zeros((10,))
delete_sample = np.argwhere(theta == 0)
print(delete_sample)
print(delete_sample.size)
