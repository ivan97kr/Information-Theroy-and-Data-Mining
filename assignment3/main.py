import myfun
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

#  req.1 Compute entropy of discrete random variable given it's probability mass function vector:
#
#
#  probability mass function
p = [0.2, 0.3, 0.4, 0.1]  # choosen p.m.f. vector

#  entropy computation
h1 = myfun.entropy(p)
print('Entropy of choosen p.m.f is: ', h1)  # verifico h

#  gen set of 1000 samples with equal p.m.f vector
vector_samples_len = 1000

#  set of values that r.v can take
discrete_values = [1, 2, 3, 4]

#  generate random samples with given probabilities
#  input to random.choice fun: possible values, prob of each value, num. of samples to gen
samples = np.random.choice(discrete_values, p=p, size=vector_samples_len)
# estimate pmf by calling myfun.discrete_pmf
sample_list_estimated, pmf_estimated = myfun.discrete_pmf(samples)
figure1 = plt.figure(1)

#  compute entropy of this pmf estimated
h2 = myfun.entropy(pmf_estimated)
print('Entropy of estimated p.m.f is: ', h2)
#  make the difference between two entropy
result_h = abs(h1 - h2)
print('Difference of the 2 entropy is: ', result_h)
#  plot graph of difference between two p.m.f. (a priori and estimated)
plt.stem(discrete_values, p)
plt.stem(sample_list_estimated, pmf_estimated)
plt.show()


#  req 2. Differential entropy function:
#
#
#  definition of a probability density function of a uniform random variable with parameter a = 1/4 and b = 1/2, to test
#  the differential entropy function
def pdf(x):
    a = 0.25
    b = 0.5
    if x < a or x > b:
        return 0
    if x < 0:
        return 0
    else:
        ret = 1 / (b - a)
        return ret


#  try to compute the integral of this prob density function, if it is 1, the computations are correct
#  the result are two values: the integral value and the error of it
res1 = myfun.integral_of_fun(pdf)
print('Area of a uniform p.d.f and error: ', res1)

#  compute diff entropy of this pdf
res2 = myfun.diff_entropy(pdf)
print('Differential entropy of a uniform p.d.f and error: ', res2)

# req.3 Compute difference of differential entropy of two Gaussian r.v. the first given a priori and the second
# estimated:
#
#
# gen p.d.f. of a Gaussian continuous random variable
std = 3  # standard deviation
mean = 5  # mean value


#  definition of gaussian function with mean value and std given
def gaussian(x):
    for i in range(vector_samples_len):  # take only 1000 sample of pdf
        return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))


#  try to compute differential entropy of this gaussian function
res3 = myfun.diff_entropy(gaussian)
#  see the result
print('Differential entropy of gaussian p.d.f. with mean: ', mean, ' and std: ', std, ' is: ', res3)

#  in order to estimate p.d.f with the same parameters by using kernel method:
#
#
#  gen the samples of gaussian with given std and mean by using normal function of numpy
a = np.random.normal(loc=mean, scale=std, size=vector_samples_len)
#  compute standards deviation and mean of samples by using numpy std adn mean function
samples_std = np.std(a)
samples_mean = np.mean(a)
#  compute the max and min values of all samples and 2*sigma
samples_max = max(a)
samples_min = min(a)
margin = 2 * samples_std

#  compute the optimal bandwidth suggested by kernel method
optimal_bandwidth = 1.06 * samples_std * np.power(vector_samples_len, -1 / 5)
#  gen kernel density object
kde_object = KernelDensity(kernel='tophat', bandwidth=optimal_bandwidth).fit(a.reshape(-1, 1))
#  compute domain of estimated gaussian pdf (x)
X_plot = np.linspace(samples_min - margin, samples_max + margin, vector_samples_len)[:, np.newaxis]
#  score_samples give as result the natural log density of estimated pdf
kde_LogDensity_estimate = kde_object.score_samples(X_plot)
#  elevate e to log to obtain the pdf
kde_estimate = np.exp(kde_LogDensity_estimate)

# Is important to note that kernel density estimator give as result finite number of samples of pdf estimated fit on
# the defined domain (X_plot), so to compute the differential entropy use simps function of scipy

#  compute the differential entropy of this estimated samples coming from estimated gaussian estimated pdf
res4 = myfun.diff_entropy_over_samples(kde_estimate, X_plot)
print('Differential entropy of gaussian estimated pdf is: ', res4)

#  compute the differnece between the diff. entropy of a priori gaussian pdf and estimated gaussian pdf
result = abs(res3-res4)
print('Difference of 2 diff. entropy is: ', result)
