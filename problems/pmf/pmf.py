from numpy import linalg as LA
import numpy as np
from math import *


def PMF(num_user, num_item, train, val, epsilon=0.1, lamb=0.01, maxepoch=50, num_feat=10, l_rating=1,
        u_rating=5, num_batches=9):
    """
    Ids of users and items start from one!

    epsilon: learning rate
    lamb: l2-regularizer
    num_feat : the matrix rank
    maxepoch: number of epochs
    """
    np.random.seed(1)
    momentum = 0.5
    epoch = 1
    mean_rating = np.mean(train[:, 2])

    pairs_tr = train.shape[0]  # training data
    pairs_va = val.shape[0]  # validation data
   #
   # # num_batches = int(pairs_tr / 256.0)
   #  num_batches = 9 # tal vez es mejor idea
    batch_size = pairs_tr / num_batches

    w1_M1 = 0.1 * np.random.randn(num_item, num_feat)  # movie feature vectors
    w1_P1 = 0.1 * np.random.rand(num_user, num_feat)  # User feature vectors
    w1_M1_inc = np.zeros((num_item, num_feat))
    w1_P1_inc = np.zeros((num_user, num_feat))

    for epoch in range(0, maxepoch):
        shuffled_order = np.arange(train.shape[0])
        np.random.shuffle(shuffled_order)

        for batch in range(num_batches):
            next_ = min(batch_size * (batch + 1), pairs_tr)
            batch_idx = np.arange(batch_size * batch, next_)

            batch_uID = np.array(train[shuffled_order[batch_idx], 0] - 1, dtype='int32')  # userID
            batch_itID = np.array(train[shuffled_order[batch_idx], 1] - 1, dtype='int32')  # itemID
            ratings = np.array(train[shuffled_order[batch_idx], 2], dtype='int32')  # itemID

            ratings = ratings - mean_rating  # default prediction is the mean_rating

            ###compute predictions

            pred = np.sum(np.multiply(w1_M1[batch_itID, :], w1_P1[batch_uID, :]), 1)
            rawErr = pred - ratings

            ######compute gradients
            IX_m = 2.0 * np.multiply(rawErr[:, np.newaxis], w1_P1[batch_uID, :]) \
                   + lamb * w1_M1[batch_itID, :]
            IX_p = 2.0 * np.multiply(rawErr[:, np.newaxis], w1_M1[batch_itID, :]) \
                   + lamb * w1_P1[batch_uID, :]

            dw_m = np.zeros((num_item, num_feat))
            dw_p = np.zeros((num_user, num_feat))

            loop = batch_size

            if batch_size * (batch + 1) > pairs_tr:
                loop = len(batch_itID)

            for i in range(loop):
                dw_m[batch_itID[i], :] += IX_m[i, :]
                dw_p[batch_uID[i], :] += IX_p[i, :]

            ##update with momentum

            w1_M1_inc = momentum * w1_M1_inc + epsilon * dw_m #/ float(batch_size)
            w1_M1 = w1_M1 - w1_M1_inc

            w1_P1_inc = momentum * w1_P1_inc + epsilon * dw_p #/ float(batch_size)
            w1_P1 = w1_P1 - w1_P1_inc
    ###compute validation error


    pred_out = np.sum(np.multiply(w1_P1[np.array(val[:, 0] - 1, dtype='int32')],
                                  w1_M1[np.array(val[:, 1] - 1, dtype='int32')]), axis=1)
    pred_out = pred_out + mean_rating
    pred_out[pred_out > u_rating] = u_rating
    pred_out[pred_out < l_rating] = l_rating

    rawErr = pred_out - val[:, 2]
    return -1.0 * np.sum(rawErr ** 2) / float(pairs_va)


def cross_validation(num_user, num_item, train, val, epsilon=1, lamb=0.01, maxepoch=50,
                     num_Feat=10):
    """
    train: list of training sets: [t1,t2,t3,...]
    val: list of validation sets: [v1,v2,v3,...]
    """
    n = len(train)

    error = 0
    for i in range(n):
        error += PMF(num_user, num_item, train[i], val[i], epsilon, lamb, maxepoch, num_Feat)
    return error / n


if __name__ == '__main__':
    num_user = 943
    num_item = 1682

    train = []
    val = []

    for i in range(1, 6):
        data = np.loadtxt("ml-100k/u%d.base" % i)
        test = np.loadtxt("ml-100k/u%d.test" % i)
        train.append(data)
        val.append(test)

    num_feature = 9
    # a=PMF(num_user,num_item,train[3],val[3],epsilon=0.2,lamb=0.291,maxepoch=89,num_feat=9)
    a = cross_validation(num_user, num_item, train, val, epsilon=0.2, lamb=0.29, maxepoch=89,
                         num_Feat=9)
    print sqrt(a)