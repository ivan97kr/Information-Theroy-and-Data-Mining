import math
import numpy as np
from scipy.integrate import quad
from scipy.integrate import simps


#   entropy fun:
#   computes the entropy of a discrete random variable given its probability mass function p = [p1; p2; :::; pN].
def entropy(p):
    #  lunghezza del vettore massa di probabilità
    n = len(p)
    #   var per il calcolo dell'entropia
    h = 0
    #  sommatoria che costituisce la entropy function
    for i in range(0, n):
        h = h + p[i] * math.log((1 / p[i]), 2)
    return h


#   discrete pmf estimation function:
#   given as input the vector of samples (realization of a discrete random variable)
#   return as output the ordered list of unique values present in samples vector and it's discrete probability mass
#   function
def discrete_pmf(samples):
    vector_len = len(samples)
    # np unique function with return_counts=true return first the vector of unique ordered values present in samples
    # vector and then the count of each repetition per each value
    sorted_unique_elements, count_rep = np.unique(samples, return_counts=True)
    pmf_estimated = count_rep / vector_len
    return sorted_unique_elements, pmf_estimated


# differential entropy function:
# given as input the probability density function (must be explicitly a function of x) return the differential entropy
# of it
def diff_entropy(pdf):
    # definition of integrand function in the case of differential entropy def
    def integrand(x):
        if pdf(x) == 0:  # to avoid zero value in log error
            return 0
        else:
            return pdf(x) * math.log((1 / pdf(x)), math.e)

    # integration from -inf to +inf
    result = quad(integrand, -math.inf, math.inf)
    return result


#  function that make integral between -inf and +inf of a given input function:
def integral_of_fun(pdf):
    # integration from -inf to +inf
    result = quad(pdf, -math.inf, math.inf)
    return result


#  function to calculate differential entropy given as input:
#  the vector of samples of pdf estimated and its domain x
def diff_entropy_over_samples(samples, x_plot):
    diff_entropy_samples = []
    x = x_plot.reshape(-1)  # reshape vector in order to have the same dimension of integrand

    #  apply the differential entropy formula to each sample
    for i in range(1000):
        if samples[i] == 0:
            diff_entropy_samples.append(0)
            continue
        diff_entropy_samples.append(samples[i] * math.log((1 / samples[i]), math.e))

    #  compute the integral over the given domain to complete the differential entropy measurement
    result = simps(diff_entropy_samples, x)
    return result
