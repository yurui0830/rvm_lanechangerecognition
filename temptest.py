import numpy as np
from concatenate_features import create_clip_extract_features
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import norm
from numpy.random import randint


rightlc_feat = create_clip_extract_features('rightlc')
leftlc_feat = create_clip_extract_features('leftlc')
lk_feat = create_clip_extract_features('lk')
# balance the dataset
leftlc_feat = leftlc_feat[0:100, :]

# total number of right lc
n_rlc = np.size(rightlc_feat, 0) # 68
# generate labels for right lc
t_rlc = np.zeros((3, n_rlc))
t_rlc[0] = 1
# number of right lc in one fold
r_set = int(n_rlc/3)
# total number of left lc
n_llc = np.size(leftlc_feat, 0) # 213
# generate labels for left lc
t_llc = np.zeros((3, n_llc))
t_llc[1] = 1
# number of left lc in one fold
l_set = int(n_llc/3)
# total number of lane keeping
n_lk = np.size(lk_feat, 0) # 112
# generate labels for lane keeping
t_lk = np.zeros((3, n_lk))
t_lk[2] = 1
# number of lane keeping in one fold
lk_set = int(n_lk/3)

# initialize tp to store true-positive rates, conf to store confusion matrices
tp = np.zeros((3,1))
conf = np.zeros((3,3), dtype=int)
for fold in range(1, 1):
    if fold == 1:
        X = np.concatenate((rightlc_feat[0:r_set], leftlc_feat[0:l_set], lk_feat[0:lk_set]))
        y = np.concatenate((t_rlc[:, 0:r_set], t_llc[:, 0:l_set], t_lk[:, 0:lk_set]), axis=1)
    elif fold == 2:
        X = np.concatenate((rightlc_feat[r_set:], leftlc_feat[l_set:], lk_feat[lk_set:]))
        y = np.concatenate((t_rlc[:, r_set:], t_llc[:, l_set:], t_lk[:, lk_set:]), axis=1)
    elif fold == 3:
        X = np.concatenate((rightlc_feat[r_set:r_set*2], leftlc_feat[l_set:l_set*2], lk_feat[lk_set:lk_set*2]))
        y = np.concatenate((t_rlc[:, r_set:r_set*2], t_llc[:, l_set:l_set*2], t_lk[:, lk_set:lk_set*2]), axis=1)
    elif fold == 4:
        X = np.concatenate((rightlc_feat[0:r_set], rightlc_feat[r_set*2:], leftlc_feat[0:l_set],
                                        leftlc_feat[l_set * 2:], lk_feat[0:lk_set], lk_feat[lk_set*2:]))
        y = np.concatenate((t_rlc[:, 0:r_set], t_rlc[:, r_set*2:], t_llc[:, 0:l_set], t_llc[:, l_set*2:],
                                       t_lk[:, 0:lk_set], t_lk[:, lk_set*2:]), axis=1)
    elif fold == 5:
        X = np.concatenate((rightlc_feat[r_set*2:], leftlc_feat[l_set*2:], lk_feat[lk_set*2:]))
        y = np.concatenate((t_rlc[:, r_set*2:], t_llc[:, l_set*2:], t_lk[:, lk_set*2:]), axis=1)
    elif fold == 6:
        X = np.concatenate((rightlc_feat[0:r_set*2], leftlc_feat[0:l_set*2], lk_feat[0:lk_set*2]))
        y = np.concatenate((t_rlc[:, 0:r_set*2], t_llc[:, 0:l_set*2], t_lk[:, 0:lk_set*2]), axis=1)
X = np.zeros((4, 5))
y = np.zeros((2, 4))
y[0, 0:2] = 1
y[1, 2:] = 1
X[0] = [2, 3, 6, -4, 11]
X[1] = [4, 6, 12, 0, 22]
X[2] = [-4, -6, -13, -11, -22]
X[3] = [-1, -2, -5, 9, -9]

k = pairwise_kernels(X, X, metric='linear')
i = np.argmin(np.sum((X.T @ y.T)**2, 1)/np.sum(X ** 2, 0))
print(i)
