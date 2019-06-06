import numpy as np

theta = np.arange(0, 10)
print(theta)
active_sample = np.zeros((10,), dtype=bool)
active_sample[5:] = True
print(theta[active_sample])
print(np.min(theta[active_sample]))
delete_sample = np.argwhere(theta == np.min(theta[active_sample]))[0][0]
print(delete_sample)