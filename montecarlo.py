import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

"""
Monte Carlo Simulation from Empirical Distributions

Function 1 ecdf: compute empirical cumulative distribution function (ecdf) using existing data
Function 2 generate_rv: use random unif(0,1) and inverse transform method to select random variate from ecdf
Function 3 simulate_rv: simulate a user specified number of RVs
Function 4 replicate: generate user defined runs of simulation 
"""

# fake data to work with
x = np.floor(np.exp(np.random.normal(2.5,0.25,10000)) + 0.5)
sns.distplot(x)
plt.show()

# function 1: get the empirical cumulative distribution
def ecdf(data):
    """
    takes a 1-D array and calculates the cumulative distribution value from {0,1}

    arguments
    -----------
    data: a 1-d array or list

    output
    -----------
    a tuple containing X (the input data) and Y (the cumulative percentile) values
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return(x,y)

# plot ecdf
x_ecdf = ecdf(x)

# sns.lineplot(x_ecdf[0], x_ecdf[1])
# plt.show()

# function 2: generate RV from ECDF
def generate_rv(ecdf):
    """
    takes a UNIF(0,1) RV and generates an RV from the ECDF

    arguments
    -----------
    ecdf: the output of the ecdf function

    output
    -----------
    a single random variate
    """
    u = np.random.uniform(0,1,1)[0]
    rv = ecdf[0][np.where(ecdf[1] >= u)[0]][0]
    return(rv)

# function 3: Simulate using RVs
def simulate_rv(ecdf, n = 100):
    """
    generates a sequence of RVs

    arguments
    -----------
    ecdf: the output of the ecdf function
    n: the length of simulation
    """
    rvs = [generate_rv(ecdf) for _ in range(n)]
    return(rvs)

# # plot results: histogram
# rv_results = simulate_rv(x_ecdf, n = 100)

# sns.distplot(rv_results)
# plt.show()

# # plot results: ecdf of RVs
# rv_ecdf = ecdf(rv_results)
# sns.lineplot(rv_ecdf[0], rv_ecdf[1])
# plt.show()

# plot results: compare ecdfs
# df = pd.DataFrame({'X': x_ecdf[0], 'Y': x_ecdf[1], 'Source': 'Orig'})
# sim_df = pd.DataFrame({'X': rv_ecdf[0], 'Y': rv_ecdf[1], 'Source': 'Sim'})

# df = df.append(sim_df)

# sns.lineplot(x = 'X', y = 'Y', data = df, hue='Source', style='Source')
# plt.show()

# function 4: replicate
def replicate(ecdf, n = 1000, replications=100, quantiles = []):
    """
    designed to capture point statistics from user-defined number of sim runs

    arguments
    ----------------
    ecdf: the empircal cumulative distribution function of the variable of interest
    n: the length of each simulation replication
    replications: the number of times to replicate the simulation
    quantiles: a list containing any quantiles of interest in the output
                default quantiles = [50, 75, 90, 95, 99]
                additional quantiles can be added

    point statistics
    ----------------
    mean
    standard deviation
    variance
    quantiles

    output
    ----------------
    dictionary containing replication number and point statistics
    """
    
    # create dataframe to hold results
    quantile_vals = list(set([50, 75, 90, 95, 99] + quantiles))
    quantile_vals.sort()
    quantile_names = ['quantile' + str(q) for q in quantile_vals]
    colnames = ['replication', 'length', 'mean', 'var', 'stddev'] + quantile_names
    results = {key : [] for key in colnames}

    # run replications and capture statistics
    for i in range(replications):
        results['replication'].append(i)
        X = simulate_rv(ecdf, n)
        results['mean'].append(np.mean(X))
        results['stddev'].append(np.std(X))
        results['var'].append(np.var(X))
        # quantiles
        for j,k in zip(quantile_names, quantile_vals):
            results[j].append(np.percentile(X, k))

    results['length'].append(list(np.repeat(n,replications)))
    return results

