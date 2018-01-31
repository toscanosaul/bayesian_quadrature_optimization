from __future__ import absolute_import

import numpy as np
import os

from problems.citi_bike_mt.simulationPoissonProcessNonHomogeneous_2 import *

n1=4
n2=1

nDays=365


nSets=4

fil="poissonDays.txt"
fil="problems/citi_bike_mt/" + fil
poissonParameters=np.loadtxt(fil)

###readData

poissonArray = [[] for i in xrange(nDays)]
exponentialTimes = [[] for i in xrange(nDays)]

for i in xrange(nDays):
    fil = "daySparse" + "%d" % i + "ExponentialTimesNonHom.txt"
    fil2 = os.path.join("problems/citi_bike_mt/SparseNonHomogeneousPP2", fil)
    poissonArray[i].append(np.loadtxt(fil2))

    fil = "daySparse" + "%d" % i + "PoissonParametersNonHom.txt"
    fil2 = os.path.join("problems/citi_bike_mt/SparseNonHomogeneousPP2", fil)
    exponentialTimes[i].append(np.loadtxt(fil2))

numberStations=329
Avertices=[[]]
for j in range(numberStations):
    for k in range(numberStations):
	    Avertices[0].append((j,k))

with open('problems/citi_bike_mt/json.json') as data_file:
    data = json.load(data_file)


f = open('problems/citi_bike_mt/' + str(4)+"-cluster.txt", 'r')
cluster=eval(f.read())
f.close()

bikeData=np.loadtxt("problems/citi_bike_mt/bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)



TimeHours=4.0
numberBikes=6000

poissonParameters*=TimeHours

###upper bounds for X
upperX=np.zeros(n1)
temBikes=bikeData[:,2]
for i in xrange(n1):
    temp=cluster[i]
    indsTemp=np.array([a[0] for a in temp])
    upperX[i]=np.sum(temBikes[indsTemp])

#n_samples=5
def toy_example(n_samples, x):
    """


    :param x: [float, float, int, int, int]
    :return: [float]
    """
    ind = None
    simulations = np.zeros(n_samples)
    # w is between 650 and 8900
    w = [int(x[-1]) + 650]

    x = x[0: -1]
    x = [int(i) for i in x]
    x.append(numberBikes - np.sum(x))
    x = np.array(x)
    for i in range(n_samples):
        simulations[i] = unhappyPeople(TimeHours, w, x, nSets,
                           data, cluster, bikeData, poissonParameters, nDays,
                           Avertices, poissonArray, exponentialTimes, randomSeed=ind)

    return [np.mean(simulations), float(np.var(simulations)) / n_samples]

##weights of w
def computeProbability(w,parLambda,nDays):
    probs=poisson.pmf(w,mu=np.array(parLambda))
    probs*=(1.0/nDays)
    return np.sum(probs)

L=650
M=8900
wTemp=np.array(range(L,M))
probsTemp=np.zeros(M-L)
for i in range(M-L):
    probsTemp[i]=computeProbability(wTemp[i],poissonParameters,nDays)


def simulatorW(n, ind=False):
    """Simulate n vectors w

       Args:
          n: Number of vectors simulated
    """
    wPrior = np.zeros((n, n2))
    indexes = np.random.randint(0, nDays, n)
    for i in range(n):
        for j in range(n2):
            wPrior[i, j] = np.random.poisson(poissonParameters[indexes[i]], 1)
    if ind:
        return wPrior, indexes
    else:
        return wPrior

import multiprocessing as mp

g=unhappyPeople

def g2(x,w,day,i):
    return g(TimeHours,w,x,nSets,
                         data,cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,day,i)

def integrate_toy_example(x):
    """Estimate g(x)=E(f(x,w,z))

       Args:
          x
          N: number of samples used to estimate g(x)
    """
    x = [int(i) for i in x]
    x.append(numberBikes - np.sum(x))
    x = np.array(x)
    N = 1000
    estimator = N
    W, indexes = simulatorW(estimator, True)
    result = np.zeros(estimator)
    rseed = np.random.randint(1, 4294967290, size=N)
    pool = mp.Pool()
    jobs = []
    for j in range(estimator):
        job = pool.apply_async(g2, args=(x, W[j, :], indexes[j], rseed[j],))
        jobs.append(job)
    pool.close()  # signal that no more data coming in
    pool.join()  # wait for all the tasks to complete

    for i in range(estimator):
        result[i] = jobs[i].get()

    return [np.mean(result), float(np.var(result)) / estimator]



def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(n_samples, *params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)