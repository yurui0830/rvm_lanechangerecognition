from rvm_model_multiclass import MultiClass_RVM
import numpy as np
from data_prepare import create_clip_extract_features, rescale, centering
from sklearn.preprocessing import normalize


def cross_validation(x, y, cv: int=5):
    # n_sampleL how many samples in the dataset; n_sample_fold: how many samples in each fold
    n_sample = x.shape[0]
    n_sample_fold = n_sample//cv
    # initialize variables
    # sample_list: use a number to label each sample
    sample_list = np.arange(0, n_sample)
    fold_old = np.arange(0, n_sample)
    accuracy = np.zeros([cv, ])
    confusion = np.zeros([3, 3])
    clf = MultiClass_RVM(max_iter=500)
    for i in range(cv):
        # fold_old: samples which have not been selected for the test set
        # fold: samples in testing set during this fold
        # fold_train: samples in training set during this fold
        if i < cv-1:
            fold = np.random.choice(fold_old, n_sample_fold, replace=False)
        else:
            fold = fold_old
        fold_train = np.setdiff1d(sample_list, fold)
        # extract training and testing set
        x_test = x[fold]
        y_test = y[fold]
        x_train = x[fold_train]
        y_train = y[fold_train]
        # update fold_old, remove the samples used during this fold
        fold_old = np.setdiff1d(fold_old, fold)
        clf.fit(x_train, y_train)
        accuracy[i], conf_temp = clf.score(x_test, y_test)
        confusion = confusion + conf_temp
    return accuracy, confusion


x_r = create_clip_extract_features('rightlc')
x_l = create_clip_extract_features('leftlc')
x_lk = create_clip_extract_features('lk')
# balance the dataset
x_l = x_l[0:68, :]
x_lk = x_lk[0:100, :]

x = np.concatenate((x_r, x_l, x_lk))
x = rescale(x, np.array([5, 0.1, 5, 0.1, 10, 10]))
y = np.zeros([x.shape[0], 3])
y[0:68, 0] = [1] * 68
y[68:136, 1] = [1] * 68
y[136:, 2] = [1] * 100

acc, conf = cross_validation(x, y, cv=5)
print(acc, conf)
