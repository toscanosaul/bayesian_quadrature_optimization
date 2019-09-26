from __future__ import absolute_import

from scipy import special
import numpy as np
from scipy.stats import gamma


def simulation(x, runlength, n_customers, n_products, cost, sell_price, mu=1.0, sum_exp=None,
               set_sum_exp=None, seed=None, util_product=0.5):
    """
    See http://simopt.org/wiki/images/e/e8/DynamicSubstitution.pdf
    :param x: ([int]) inventory levels
    :param runlength: (int) replications
    :param n_customers: (int) number of customers
    :param n_products: (int) number of products
    :param mu: (float) See the description of the problem
    :param cost: ([float]) Cost of the products
    :param sell_price: ([float]) Sell price of the products
    :param sum_exp: ([float]) The value of the sum of the exponential vector over the n_customers
        associated to the Gumbel distributions of the preferences.
    :param set_sum_exp: ([[float, float]]) The value of the sum of the exponential vector over the
        n_customers associated to the Gumbel distributions of the preferences must be in this set.
    :param seed: int
    :return: [(float) mean of the profit, (float) variance of the profit]
    """



    if seed is not None:
        np.random.seed(seed)

    cost = np.array(cost)
    sell_price = np.array(sell_price)

    n = n_products
    T = n_customers
    u = util_product * np.ones(n)  # product constant

    x = np.array([x])

    if sum_exp is None and set_sum_exp is None:

        gumbel = np.random.gumbel(mu * special.psi(1.0), mu, [n, runlength, T])
    elif sum_exp is not None:
        gumbel_array = np.zeros([n, runlength, T])
        # See p.243 of Simulation by Sheldon Ross
        for j in xrange(runlength):
            for product in xrange(n):
                exponential = rejection_sampling_cond_exponential(1, T, sum_exp[product], mu)[0, :]
                gumbel = np.zeros(T)
                for i in xrange(T):
                    gumbel[i] = (1.0 - np.exp(- exponential[i] * mu))
                    gumbel[i] = mu * special.psi(1.0) - mu * np.log(-np.log(gumbel[i]))
                gumbel_array[product, j, :] = gumbel
        gumbel = gumbel_array
    elif set_sum_exp is not None:
        gumbel_array = np.zeros([n, runlength, T])
        for j in xrange(runlength):
            for product in xrange(n):
                exponential = rejection_sampling_cond_set_exponential(1, T, set_sum_exp[product], mu)[0,:]
                # upper = set_sum_exp[product][1]
                # lower = set_sum_exp[product][0]
                # if upper == np.inf:
                #     up = max(T * (1.0 / mu) + T * 10.0 * (mu ** -2), lower + 1)
                # else:
                #     up = upper
                # start = np.zeros(T)
                # for i in xrange(T):
                #     start[i] = np.random.uniform(lower / T, up / T, 1)
                # exponential = \
                #     gibbs_sampler(start, T, set_sum_exp[product], mu, 500, 5, 1)[0, :]
                gumbel = np.zeros(T)
                for i in xrange(T):
                    gumbel[i] = (1.0 - np.exp(- exponential[i] * mu))
                    gumbel[i] = mu * special.psi(1.0) - mu * np.log(-np.log(gumbel[i]))
                gumbel_array[product, j, :] = gumbel
        gumbel = gumbel_array


    # Determine Utility Function
    utility = np.zeros((n, runlength, T))
    for i in xrange(n):
        utility[i, :, :] = u[i] + gumbel[i, :, :]

    initial = np.repeat(x, runlength, axis=0)
    inventory = initial.copy()

    for j in xrange(T):
        for k in xrange(runlength):
            available = np.where(inventory[k,:]>0)
            if len(available[0]) == 0:
                continue
            decision = utility[available[0],k,j]
            maxVal = max(decision)
            if maxVal > 0:
                index = np.where(utility[:,k,j] == maxVal)[0][0]
                inventory[k, index] -= 1

    # Compute daily profit
    numSold =initial - inventory
    unitProfit=sell_price-cost
    singleRepProfit=np.dot(numSold, unitProfit)
    singleRepProfit -= np.dot(inventory, cost)
    fn = np.mean(singleRepProfit)
    FnVar = np.var(singleRepProfit)/runlength

    return [fn, FnVar]


def gibss_sampler_exponential(starting, n, limit, lamb):
    """
    See p. 279 of Simulation by Sheldon Ross
    :param starting:
    :param n: (int) number of random variables
    :param limit: [float, float]
    :param lamb: parameter of exponential
    :return:
    """
    sample = starting

    for index in xrange(n):
        u = np.random.uniform(0, 1, 1)
        sum = 0
        for i in xrange(n):
            if i != index:
                sum += sample[i]
        a1 = max(limit[0] - sum, 0)
        a2 = limit[1] - sum
        cte = np.exp(-lamb * a1) - np.exp(-lamb * a2)
        x = -np.log(-u * cte + np.exp(-lamb * a1)) / lamb

        sample[index] = x
    return sample

def gibbs_sampler(starting, n, limit, lamb, burning, thinning, n_samples):
    for i in xrange(burning):
        starting = gibss_sampler_exponential(starting, n, limit, lamb)
    samples = np.zeros((n_samples, n))
    for j in xrange(n_samples):
        for l in xrange(thinning):
            starting = gibss_sampler_exponential(starting, n, limit, lamb)
        samples[j, :] = starting
    return samples


def rejection_sampling_cond_exponential(n_samples, T, sum_exp, mu):
    n = 0
    samplings = []
    cte = gamma.pdf(sum_exp, T, loc=0, scale=1 / mu)
    M = 1.0 / cte
    while (n < n_samples):
        u = np.random.uniform(0, 1, 1)
        exponential = np.random.exponential(1 / mu, T - 1)
        if exponential.sum() > sum_exp:
            continue
        ratio = (1/mu)* np.exp(-(sum_exp - exponential.sum())) / cte
        if u < ratio / M:
            exponential = np.concatenate([exponential, [sum_exp - exponential.sum()]])
            samplings.append(exponential)
            n += 1
    return np.array(samplings)

def rejection_sampling_cond_set_exponential(n_samples, T, set_exp, mu):
    n = 0
    samplings = []
    while (n < n_samples):
        exponential = np.random.exponential(1 / mu, T)
        if exponential.sum() > set_exp[1] or exponential.sum() < set_exp[0]:
            continue
        samplings.append(exponential)
        n += 1
    return np.array(samplings)



