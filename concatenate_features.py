import numpy as np
import csv

def create_clip_extract_features(filename, slot = '1s'):

    """
    function create_clip_extract_features(filename, slot='1s')

    :param
        filename(str: 'rightlc', 'leftlc' or 'lk'): filename of the .csv file
        slot(str: '1s', '0.5s', or 'whole'): how long of the procedure is preserved in the training set

    :return
        features(array: n_samples * n_features): n_samples = n_sequence * time steps inside each sequence

    """

# open .csv file, store all the data in rows(list)
    with open(filename+'_101.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader_1 = csv.reader(csvfile)
        # store data in rows
        rows_1 = [row for row in reader_1]
    csvfile.close()
    with open(filename+'_i80.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader_2 = csv.reader(csvfile)
        # store data in rows
        rows_2 = [row for row in reader_2]
    csvfile.close()

# extract n_sequences, store in sequence+'i'
    # headings(list): categories of the data (omit veh_id)
    headings = rows_1[0][1:]
    # n_sequence(int): total number of sequences
    n_sequence_1 = int(rows_1[-1][0])
    n_sequence_2 = int(rows_2[-1][0])
    # initialize length
    length_1 = np.zeros((n_sequence_1,), dtype=int)
    length_2 = np.zeros((n_sequence_2,), dtype=int)
# extract n_sequence_1 from file_1
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence_1_' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows_1[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length_1[veh_num-1] = length_1[veh_num-1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence_1_' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length_1[veh_num-1] = length_1[veh_num-1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)
# extract n_sequence_2 from file_2
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence_2_' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows_2[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length_2[veh_num - 1] = length_2[veh_num - 1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence_2_' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length_2[veh_num - 1] = length_2[veh_num - 1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)

# extract features
    # features(array: number of total sequences*number of categories)
    features = np.empty((np.sum(length_1)+np.sum(length_2), 6))
    n_sequence = n_sequence_1 + n_sequence_2
    length = np.zeros((n_sequence,), dtype=int)
    # seq_range(array: n_sequence+1,): each element represent the start row of each sequence in the feature vector
    seq_range = np.zeros((n_sequence + 1,), dtype=int)
    i_1 = i_2 = 1
    for i in range(1, np.size(length, 0)+1):
        if i % 2 == 1 and i_1 <= n_sequence_1:
            seq_1 = 'sequence_1_' + str(i_1)
            features[seq_range[i-1]:seq_range[i-1]+length_1[i_1-1]] = globals()[seq_1][:]
            length[i-1] = length_1[i_1-1]
            i_1 = i_1 + 1
        elif i % 2 == 0 and i_2 <= n_sequence_2:
            seq_2 = 'sequence_2_' + str(i_2)
            features[seq_range[i-1]:seq_range[i-1]+length_2[i_2-1]] = globals()[seq_2][:]
            length[i-1] = length_2[i_2-1]
            i_2 = i_2 + 1
        elif i % 2 == 1 and i_1 > n_sequence_1:
            seq_2 = 'sequence_2_' + str(i_2)
            features[seq_range[i-1]:seq_range[i-1]+length_2[i_2-1]] = globals()[seq_2][:]
            length[i-1] = length_2[i_2-1]
            i_2 = i_2 + 1
        elif i % 2 == 0 and i_2 > n_sequence_2:
            seq_1 = 'sequence_1_' + str(i_1)
            features[seq_range[i-1]:seq_range[i-1]+length_1[i_1-1]] = globals()[seq_1][:]
            length[i-1] = length_1[i_1-1]
            i_1 = i_1 + 1
        seq_range[i] = np.array([np.sum(length[:i])])

# label the features by states
    # initialize label(array: n_samples,)
    label = np.zeros((np.sum(length[:]),), dtype=int)
    # label lane keeping features
    # features[3]: target_acc_y, min(features[3,seq:seq+15])>0.4: acc is higher than 0.4 m/s^2 for 1.5 second in a row
    # 0: normal driving; 1: acceleration; 2: deceleration
    if 'lk' in filename:
        # lk class only contains one state
        label[:] = 0
    # label the left lane changes
    # features[4]: ref_heading < -0.2, features[5]: yaw_rate <0/>0
    # 0: normal driving; 1: steer to left; 2: steer back
    elif 'leftlc' in filename:
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while max(features[seq:seq + n, 4]) < -0.2 and max(features[seq:seq + n, 5]) < 0 and seq + n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            for i in range(np.size(length, 0)):
                if seq_range[i] <= seq < seq_range[i + 1]:
                    break
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer
            if n > 6 and seq >= seq_range[i] + 10:
                if seq + n >= seq_range[i + 1]:
                    n = seq_range[i + 1] - 1 - seq
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq + n] = 1
                    seq = seq + n
            else:
                seq = seq + 1
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while max(features[seq:seq + n, 4]) < -0.2 and min(features[seq:seq + n, 5]) > 0 and seq + n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer back
            if n > 6:
                for i in range(np.size(length, 0)):
                    if seq_range[i] <= seq < seq_range[i + 1]:
                        if seq + n >= seq_range[i + 1]:
                            n = seq_range[i + 1] - 1 - seq
                        break
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq + n] = 2
                    seq = seq + n
            else:
                seq = seq + 1
        # label the right lane changes
        # features[4]: ref_heading > 0.2, features[5]: yaw_rate >0/<0
        # 0: normal driving; 1: steer to right; 2: steer back
    elif 'rightlc' in filename:
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while min(features[seq:seq+n, 4]) > 0.2 and min(features[seq:seq+n, 5]) > 0 and seq + n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            for i in range(np.size(length, 0)):
                if seq_range[i] <= seq < seq_range[i + 1]:
                    break
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer
            if n > 6 and seq >= seq_range[i] + 10:
                if seq + n >= seq_range[i + 1]:
                    n = seq_range[i + 1] - 1 - seq
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq + n] = 1
                    seq = seq + n
            else:
                seq = seq + 1
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while min(features[seq:seq+n, 4]) > 0.2 and max(features[seq:seq+n, 5]) < 0 and seq + n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer back
            if n > 6:
                for i in range(np.size(length, 0)):
                    if seq_range[i] <= seq < seq_range[i + 1]:
                        if seq + n >= seq_range[i + 1]:
                            n = seq_range[i + 1] - 1 - seq
                        break
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq + n] = 2
                    seq = seq + n
            else:
                seq = seq + 1

# clip the lane change/keeping procedures
    # lk class and the whole process do not need to be clipped
    if filename != 'lk' and slot != 'whole':
        # initialize variables
        start_point = np.zeros((n_sequence,), dtype=int)
        seq_range_clip = np.zeros((n_sequence + 1,), dtype=int)
        features_clip = np.zeros((50 * n_sequence, np.size(features, 1)))
        # find start point, extract part of the features and labels, update new seq_range
        for i in range(n_sequence):
            start = seq_range[i]
            stop = seq_range[i + 1]
            # find the start point by labels
            start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
            # extract features and labels
            if start_point[i] - 20 > seq_range[i]:
                seq_range_clip[i + 1] = seq_range_clip[i] + 30
                features_clip[seq_range_clip[i]:seq_range_clip[i + 1]] = features[start_point[i] - 20:start_point[i] + 10]
            else:
                seq_range_clip[i + 1] = seq_range_clip[i] + start_point[i] + 10 - seq_range[i]
                features_clip[seq_range_clip[i]:seq_range_clip[i + 1]] = features[seq_range[i]:start_point[i] + 10]
        # delete unused rows from features_new
        features_clip = np.delete(features_clip, np.s_[seq_range_clip[n_sequence]:], 0)
    else:
        features_clip = features
        seq_range_clip = seq_range

# use means and variances to replace original features
    feat_mu_sigma = np.zeros((n_sequence, 2*6))
    for i in range(n_sequence):
        feat_mu_sigma[i, 0:6] = np.mean(features_clip[seq_range_clip[i]:seq_range_clip[i+1]], axis=0)
        feat_mu_sigma[i, 6:] = np.var(features_clip[seq_range_clip[i]:seq_range_clip[i + 1]], axis=0)

# return value
    return feat_mu_sigma
