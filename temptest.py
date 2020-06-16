import numpy as np
from data_prepare import create_clip_extract_features
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import norm
from numpy.random import randint


accu = np.zeros([30])
prec = np.zeros([30])
recall = np.zeros([30])


accu[0:10] = [0.95742129, 0.92263868, 0.94527736, 0.92608696, 0.92698651, 0.92008996, 0.9113943,  0.89835082, 0.89925037, 0.89835082]
prec[0:10] = [0.99559471, 0.99090909, 1.,         0.99545455, 0.97767857, 0.98190045, 0.97285068, 0.96803653, 0.96363636, 0.95495495]
recall[0:10] = [0.97413793, 0.96888889, 0.97379913, 0.96475771, 0.96475771, 0.96875, 0.97285068, 0.95927602, 0.95495495, 0.97247706]

accu[10:20] = [0.91394303, 0.8958021,  0.92098951, 0.90449775, 0.94437781, 0.95142429, 0.90614693, 0.91589205, 0.89745127, 0.87661169]
prec[10:20] = [0.97297297, 0.95475113, 0.97309417, 0.97260274, 0.99111111, 0.99555556, 0.97716895, 0.97285068, 0.96363636, 0.95833333]
recall[10:20] = [0.96860987, 0.96788991, 0.96875,    0.95515695, 0.98237885, 0.97816594, 0.95964126, 0.97727273, 0.96803653, 0.95833333]

accu[20:] = [0.894003,   0.86881559, 0.85757121, 0.82548726, 0.84962519, 0.86701649, 0.82008996, 0.83583208, 0.81244378, 0.82713643]
prec[20:] = [0.97685185, 0.94907407, 0.95283019, 0.93269231, 0.95260664, 0.97156398, 0.92822967, 0.93364929, 0.93170732, 0.93301435]
recall[20:] = [0.96347032, 0.96244131, 0.94835681, 0.94174757, 0.95260664, 0.94907407, 0.94634146, 0.93809524, 0.92270531, 0.93301435]

f1 = 2 * prec * recall / (prec + recall)


plt.figure(1)
plt.plot(np.arange(1, 4, 0.1), prec, 'b', label='precision')
plt.plot(np.arange(1, 4, 0.1), recall, 'g', label='recall')
plt.ylim((0.9, 1))
plt.xlabel('Seconds after the Behavior Starts (second)')
plt.ylabel('Recognition Performance')
plt.legend()
plt.grid(True)
plt.figure(2)
plt.plot(np.arange(1, 4, 0.1), f1, 'b')
plt.ylim((0.9, 1))
plt.xlabel('Seconds after the Behavior Starts (second)')
plt.ylabel('Recognition Performance (F1 Score)')
plt.grid(True)
plt.show()
