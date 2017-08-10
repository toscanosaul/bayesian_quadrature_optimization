import matlab.engine
import os

eng = matlab.engine.start_matlab()
cwd = os.getcwd()
eng.addpath(cwd + '/problems/pmf/')

def PMF(num_user, num_item, train, val, epsilon=50, lamb=0.01, maxepoch=50, num_feat=10,
        l_rating=1, u_rating=5):

    # train = [list(train[i,:]) for i in xrange(train.shape[0])]
    # val = [list(val[i, :]) for i in xrange(val.shape[0])]
    #
    # train = matlab.double(train)
    # val =  matlab.double(val)

    return eng.pmf(
        num_user, num_item, train, val, epsilon, lamb,  maxepoch, num_feat, l_rating, u_rating)
