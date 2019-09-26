from __future__ import absolute_import

from scipy import special
import numpy as np
from scipy.stats import gamma


def cdf(x, mu):
    """
    cdf of a gumbel based on simpot document
    :param x:
    :param mu:
    :return:
    """
    gamma = -1.0 * special.psi(1)
    return np.exp(-np.exp(-((x / mu) + gamma)))


def probability_1(utility, mu):
    """
    probability of E1 > E2 > E3, where E3 is not to buy anything
    or E2 > E1 > E3. (E3=0)
    :return:
    """
    cte1 = (1 - cdf(-utility, mu))
    gamma = -1.0 * special.psi(1)
    cte = 0.5 * (1.0 - np.exp(-2.0 * np.exp(-((-utility / mu)+gamma))))

    return cte1 - cte

def probability_2(utility, mu):
    """
    probability of E1 > E3 > E2, where E3 is not to buy anything
    or E2 > E3 > E1 (E3=0)
    :return:
    """
    return cdf(-utility, mu) * (1 - cdf(-utility, mu))

def probability_3(utility, mu):
    """
    probability of E3 > E1 > E2, where E3 is not to buy anything
    or E3 > E2 > E1 (E3=0)
    :return:
    """
    return 1 - 2.0 * probability_2(utility, mu) - 2.0 * probability_1(utility, mu)


utility = 0.5
mu = 1.0

p1 =probability_1(utility, mu)
p2 = probability_1(utility, mu)
p3 = probability_2(utility, mu)
p4 = probability_2(utility, mu)
p5 = probability_3(utility, mu)




def conditional_simulation(x, runlength, n_customers, n_products, cost, sell_price, mu=1.0,
                           sum_exp=None, set_sum_exp=None, seed=None, relative_order=None):
    """
    relative_order = [n1, n2, n3, n4]
    n1: E1 > E2 > E3, E2>E1>E3, E1 > E3> E2, E2 > E3 > E1
    (E3=0)
    E3 is no product
    """
    if seed is not None:
        np.random.seed(seed)

    N = n_customers
    solutions = []

    for l in range(runlength):

        # permutations
        z = list(range(N))
        for i in range(N):
            integer = np.random.randint(N - i, size=1)[0] + i
            tmp = z[i]
            z[i] = z[integer]
            z[integer] = tmp

        initial = [int(y) for y in x]

        inventory = list(initial)

        for j in z:
            if j < relative_order[0]:
                """
                we are in E1 > E2 > no_product
                """
                if inventory[0] > 0:
                    inventory[0] -= 1
                elif inventory[1] > 0:
                    inventory[1] -= 1
            elif j >= relative_order[0] and j < np.sum(relative_order[0: 2]):
                """
                we are in E2> E1 > no_product
                """
                if inventory[1] > 0:
                    inventory[1] -= 1
                elif inventory[0] > 0:
                    inventory[0] -= 1
            elif j >= np.sum(relative_order[0: 2]) and j < np.sum(relative_order[0: 3]):
                """
                we are in E1 > no_product> E2
                """
                if inventory[0] > 0:
                    inventory[0] -= 1
            elif j >= np.sum(relative_order[0: 3]) and j < np.sum(relative_order[0: 4]):
                """
                we are in E2 > no_product > E1
                """

                if inventory[1] > 0:
                    inventory[1] -= 1

        numSold = np.array(initial) - np.array(inventory)

        cost = np.array(cost)
        sell_price = np.array(sell_price)
        unitProfit = sell_price - cost

        singleRepProfit = np.dot(numSold, unitProfit)
        solutions.append(singleRepProfit)
    return np.mean(solutions), np.var(solutions) / float(runlength)


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
    x = [int(y) for y in x]

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
    FnVar = np.var(singleRepProfit)/float(runlength)

    print (fn, FnVar)
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



