#!/usr/local/bin/python3
"""
Code by: nakopa,pvajja,rparvat
Authors: Naga Anjaneyulu , Prudhvi Vajja , Rutvik Parvataneni
"""
import numpy as np
import time
from scipy import stats

# Load Data
def read_file(filename):
    f = open(filename, "r")
    train_data = []
    lines = f.readlines()
    for x in lines:
        l = x.split()
        train_data.append(l[1:])
    f.close()
    return np.array(train_data)

def train_model(arg1,arg2):
    train_data = open(arg1, 'r')
    lines = train_data.readlines()
    new_file = open(arg2, "w")
    new_file.writelines(lines)

def test_model(arg1,arg2):
    train_data = read_file(arg2)
    test_data = read_file(arg1)
    Train = train_data
    Train = Train.astype(int)
    y_train, x_train = Train[:, 0], Train[:, 1:]
    x_train = np.array(x_train)
    test_data = read_file(arg1)
    Test = test_data
    Test = Test.astype(int)
    s = time.time()
    n = 10
    error = 0
    y_pred = []

    # Iterate over all Test Data Points
    for test in Test:
        y = np.array(test[1:])
        y = y.reshape(1,192)

        # calculate the Ecludian Distance for each data point.
        z = np.sum(np.square(x_train - y),axis = 1)

        # Combine the labels with the distances
    #     l = tuple(zip(y_train, z))
        l = np.column_stack((y_train, z))

        # Sorted the nearest distances
        l = np.array(sorted(l, key = lambda x: x[1]))

        # Took K nearest Neighbours
        k = l[:n]

        # Unzipped the labeled Seperately
    #     k = list(zip(*k))
        k = np.hsplit(k, 2)[0]

        # Took the max repating element inthe neighbours
        m = stats.mode(k)
        # print(m)
        y_pred.append(m[0])
    return y_pred

