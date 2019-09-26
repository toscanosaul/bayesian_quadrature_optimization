from __future__ import absolute_import

from scipy import special
import numpy as np


def simulation(x, runlength, n_customers, n_products, cost, sell_price, mu=1.0, sum_gumbel=None,
               set_gumbel=None, seed=None, util_product=0.5):
    """
    See http://simopt.org/wiki/images/e/e8/DynamicSubstitution.pdf
    :param x: ([int]) inventory levels
    :param runlength: (int) replications
    :param n_customers: (int) number of customers
    :param n_products: (int) number of products
    :param mu: (float) See the description of the problem
    :param cost: ([float]) Cost of the products
    :param sell_price: ([float]) Sell price of the products
    :param sum_gumbel: ([float]) The value of the sum of the Gumbel vector associated to the
        products over the n_customers
    :param set_gumbel: ([[float, float]]) The sum of the Gumbel vector associated to the
        products over the n_customers must be in this set.
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

    if sum_gumbel is None and set_gumbel is None:
        gumbel = np.random.gumbel(mu * special.psi(1.0), mu, [n, runlength, T])
    elif sum_gumbel is not None:
        sum_gumbel = np.array([sum_gumbel])
        gumbel = np.random.gumbel(mu * special.psi(1.0), mu, [n, runlength, T-1])
        total_sum = np.repeat(sum_gumbel, runlength, axis=0).transpose()
        additional = total_sum - gumbel.sum(axis=2)
        additional = additional.reshape((n, runlength, 1))
        gumbel = np.concatenate((gumbel, additional), axis=2)
    elif set_gumbel is not None:
        gumbel_array = np.zeros([n, runlength, T])
        for j in xrange(runlength):
            for product in xrange(n):
                constraint = set_gumbel[product]
                gumbel = np.random.gumbel(mu * special.psi(1.0), mu, T)
                g_sum = gumbel.sum()
                while (g_sum >= constraint[1] or g_sum < constraint[0]):
                    gumbel = np.random.gumbel(mu * special.psi(1.0), mu, T)
                    g_sum = gumbel.sum()
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
            decision = utility[available[0],k,j]
            maxVal = max(decision)
            if maxVal > 0:
                index = np.where(utility[:,k,j] == maxVal)[0][0]
                inventory[k, index] -= 1

    # Compute daily profit
    numSold =initial - inventory
    unitProfit=sell_price-cost
    singleRepProfit=np.dot(numSold,unitProfit)
    fn = np.mean(singleRepProfit)
    FnVar = np.var(singleRepProfit)/runlength

    return [fn, FnVar]
