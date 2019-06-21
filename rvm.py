from skrvm import RVR
from rvm_model import Base_RVM
from rvm_model_upgrade import OneClass_RVM
import numpy as np
from concatenate_features import create_clip_extract_features

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
for fold in range(3):
    # split the training set and the test set for each fold
    if fold == 0:
        Xtest = np.concatenate((rightlc_feat[0:r_set], leftlc_feat[0:l_set], lk_feat[0:lk_set]))
        ytest = np.concatenate((t_rlc[:, 0:r_set], t_llc[:, 0:l_set], t_lk[:, 0:lk_set]), axis=1)
        Xtrain = np.concatenate((rightlc_feat[r_set:], leftlc_feat[l_set:], lk_feat[lk_set:]))
        ytrain = np.concatenate((t_rlc[:, r_set:], t_llc[:, l_set:], t_lk[:, lk_set:]), axis=1)
    elif fold == 1:
        Xtest = np.concatenate((rightlc_feat[r_set:r_set*2], leftlc_feat[l_set:l_set*2], lk_feat[lk_set:lk_set*2]))
        ytest = np.concatenate((t_rlc[:, r_set:r_set*2], t_llc[:, l_set:l_set*2], t_lk[:, lk_set:lk_set*2]), axis=1)
        Xtrain = np.concatenate((rightlc_feat[0:r_set], rightlc_feat[r_set*2:], leftlc_feat[0:l_set],
                                leftlc_feat[l_set * 2:], lk_feat[0:lk_set], lk_feat[lk_set*2:]))
        ytrain = np.concatenate((t_rlc[:, 0:r_set], t_rlc[:, r_set*2:], t_llc[:, 0:l_set], t_llc[:, l_set*2:],
                                t_lk[:, 0:lk_set], t_lk[:, lk_set*2:]), axis=1)
    elif fold == 2:
        Xtest = np.concatenate((rightlc_feat[r_set*2:], leftlc_feat[l_set*2:], lk_feat[lk_set*2:]))
        ytest = np.concatenate((t_rlc[:, r_set*2:], t_llc[:, l_set*2:], t_lk[:, lk_set*2:]), axis=1)
        Xtrain = np.concatenate((rightlc_feat[0:r_set*2], leftlc_feat[0:l_set*2], lk_feat[0:lk_set*2]))
        ytrain = np.concatenate((t_rlc[:, 0:r_set*2], t_llc[:, 0:l_set*2], t_lk[:, 0:lk_set*2]), axis=1)
    # initialize prob to store regression results
    prob = np.zeros((3,np.size(Xtest,0)))
    # model training
    clf = RVR()
    #clf = Base_RVM()
    #clf = OneClass_RVM()
    for cls in range(3):
        clf.fit(Xtrain, ytrain[cls])
        prob[cls] = clf.predict(Xtest)
    # pick the highest probability, extract the real labels from the test data
    result = np.argmax(prob, axis=0)
    t = np.argmax(ytest,axis=0)
    # calculate tp rates for each fold
    tp[fold] = 1 - np.count_nonzero(result-t)/np.size(Xtest,0)
    # calculate confusion matrices
    for n in range(np.size(Xtest,0)):
        conf[t[n], result[n]] = conf[t[n], result[n]]+1
    #print(prob)
accuracy = np.mean(tp)
print(accuracy)
print(conf)
