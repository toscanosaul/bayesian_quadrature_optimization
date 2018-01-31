from __future__ import absolute_import

import numpy as np

from problems.citi_bike_mt.simulationPoissonProcess import *

TimeHours=4.0
numberBikes=6000

fil="problems/citi_bike_mt/2014-05PoissonParameters.txt"
nSets=4
A,lamb=generateSets(nSets,fil)

n1 = 4
n2 = 4
parameterSetsPoisson=np.zeros(n2)
for j in range(n2):
    parameterSetsPoisson[j]=np.sum(lamb[j])


exponentialTimes=np.loadtxt("problems/citi_bike_mt/2014-05"+"ExponentialTimes.txt")
with open ('problems/citi_bike_mt/json.json') as data_file:
    data=json.load(data_file)

f = open("problems/citi_bike_mt/"+str(4)+"-cluster.txt", 'r')
cluster=eval(f.read())
f.close()

bikeData=np.loadtxt("problems/citi_bike_mt/bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)

range_for_w = []
number_points_dimension = []
for i in range(n2):
    upper = poisson.ppf(0.52, parameterSetsPoisson[i])
    lower = poisson.ppf(0.48, parameterSetsPoisson[i])
    range_for_w.append(range(int(lower), int(upper)+1))
    number_points_dimension.append(int(upper) + 1 - int(lower))

seq_tasks = np.cumprod(number_points_dimension)
number_tasks = seq_tasks[-1]


weights = []

def simulatorW(n):
    """Simulate n vectors w

       Args:
          n: Number of vectors simulated
    """
    wPrior = np.zeros((n, n2))
    for i in range(n2):
        wPrior[:, i] = np.random.poisson(parameterSetsPoisson[i], n)
    return wPrior

#n_samples=5
def toy_example(n_samples, x):
    """


    :param x: [float, float, int, int, int]
    :return: [float]
    """
    w = int(x[-1])
    new_w = np.zeros(n2)

    for i in range(n2):
        if n2 - 2 - i > -1:
            divisor = seq_tasks[n2 - 2 - i]
        else:
            divisor = 1
        index_entry = w / divisor

        new_w[n2 - 1 - i] = range_for_w[n2 - 1 - i][index_entry]
        w = w % divisor
    x = x[0: -1]
    x = [int(i) for i in x]
    x.append(numberBikes - np.sum(x))
    x = np.array(x)
    simulations = np.zeros(n_samples)
    for i in range(n_samples):
        simulations[i] = unhappyPeople(TimeHours,new_w,x,nSets,lamb,A,"2014-05",exponentialTimes,
                         data,cluster,bikeData)

    return [np.mean(simulations), float(np.var(simulations))/float(n_samples)]

def integrate_toy_example(x, N=100):
    """

    :param x: [float, float]
    :return: [float]
    """

    x = [int(i) for i in x]
    x.append(numberBikes - np.sum(x))
    x = np.array(x)

    estimator = N
    W = simulatorW(estimator)
    result = np.zeros(estimator)
    for i in range(estimator):
        result[i] = unhappyPeople(
            TimeHours, W[i, :], x, nSets, lamb, A, "2014-05", exponentialTimes,
            data, cluster, bikeData)

    return [np.mean(result), float(np.var(result)) / estimator]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)