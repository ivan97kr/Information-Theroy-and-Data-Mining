import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


#   discrete pmf estimation function:
#   given as input the vector of samples (realization of a discrete random variable)
#   return as output the ordered list of unique values that are in samples vector and it's discrete probability mass
#   function
def discrete_pmf(samples):
    vector_len = len(samples)
    # np unique function with return_counts=true return first the vector of unique ordered values that are in samples
    # vector and then the count of each repetition per each value
    sorted_unique_elements, count_rep = np.unique(samples, return_counts=True)
    pmf_estimated = count_rep / vector_len
    return sorted_unique_elements, pmf_estimated


# bayes classifier function:
# take in input a training dataset, a class labal vector and a test dataset,
# provide in output a class label vector for the test dataset
def bayes_classifier(train_dataset, train_class, test_dataset):
    # training phase1
    # compute pmf of training class label vector
    unique_class, class_pmf = discrete_pmf(train_class)
    # print in case of control
    #print('Pmf of training class label: ', class_pmf)
    #print('Shape training pmf of class label: ', class_pmf.shape)
    #print('Len training pmf of class label: ', len(class_pmf))

    # training phase2
    # divide training dataset into j (len of training class label) sub dataset and
    # estimate its probability density function (instances are realization of continuous r.v.)
    d1 = {}  # dictionary of sub_pdf values (kde result)
    d2 = {}  # dictionary of data_plot on which sub pdfs are mapped on (domain of resulting kde)
    len_sub_dataset = int(len(train_dataset) / len(class_pmf))  # len of each subdataset
    print('len sub dataset = ', len_sub_dataset)
    # split training dataset and compute pdf with gaussian kernel density estimator
    for i in range(1, len(class_pmf) + 1):
        # print in case for control
        # print('i =', i, ' indici = ', (i - 1) * len_sub_dataset, '-', len_sub_dataset * i)
        # print('Sub training dataset ', i, ' :', train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :])
        # print('Sub training dataset ', i, ' len:',len(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :]))
        # print('Sub training dataset ', i, ' shape:',train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :].shape)

        # save result of pdf of each sub dataset in a dictionary, return of pdf multivariate_fun estimator are 2 np
        # array, first is the matrix domain of features and second is the value of pdf function in each point of domain
        d1["sub_data_plot{0}".format(i)], d2["sub_pdf{0}".format(i)] = pdf_multivariate(
            train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :])

    # test phase1
    # now compute the class vector for the test dataset by applying Bayes theorem
    # test each row of test dataset and assign them to class value which maximize posteriori prob p(cj|x)...
    print('Computing class vector for test data set...')
    # per each row of test dataset find p(x|cj)*p(cj) product per each cj and then select the max
    c = []
    bayes_arg = []  # values p(x|cj)*p(cj)
    for k in range(0, len(test_dataset)):  # for each row of test dataset
        bayes_arg.clear()  # clear it at start of each row computation
        for j in range(1, len(class_pmf) + 1):  # compute p(x|cj)*p(cj) per each j
            # (is important to note that in denominator of bayes theorem is not considered because its cost over j)
            # compute first the conditional prob. p(x|cj) by using conditionl_prob fun
            conditional_probability = conditional_prob(d1["sub_data_plot{0}".format(j)], test_dataset[k],
                                                       d2["sub_pdf{0}".format(j)])
            #print('Coditional prob. value: ', conditional_probability)
            # collect into bayes_arg all j values p(x|cj)*p(cj)
            bayes_arg.append(conditional_probability * class_pmf[j - 1])
            #print('Bayes arg ciclo', j, ' : ', bayes_arg)
        # select the max take c value and append to new class vector
        max = np.argmax(bayes_arg)
        print('Max for row ', k, ' is: ', max)
        c.append(max)
    print('c1 returned from full bayes classifier   : ', c)
    print('c len: ', len(c))
    return c


# pdf multivariate function:
# given as input a matrix of data samples, compute features stats and fit gaussian model
# on these by using kernel density estimator, return as output vector of pdf values (kde_estimate) and matrix of
# variable (data_plot) on which they are mapped on (in practice the domain) ex. [x,y,w,z] per each kde_estimate value
def pdf_multivariate(data_samples):
    # compute stat of each feature take in consideration
    mean1 = data_samples[:, 0].mean()
    mean2 = data_samples[:, 1].mean()
    mean3 = data_samples[:, 2].mean()
    mean4 = data_samples[:, 3].mean()
    std1 = data_samples[:, 0].std()
    std2 = data_samples[:, 1].std()
    std3 = data_samples[:, 2].std()
    std4 = data_samples[:, 3].std()
    #print('mean: ', mean1, mean2)
    #print('std: ', std1, std2)
    # build features domain
    n_samples = 10
    xplot = np.linspace(mean1 - 2 * std1, mean1 + 2 * std1, n_samples, endpoint=True)
    yplot = np.linspace(mean2 - 2 * std2, mean2 + 2 * std2, n_samples, endpoint=True)
    zplot = np.linspace(mean3 - 2 * std3, mean3 + 2 * std3, n_samples, endpoint=True)
    wplot = np.linspace(mean4 - 2 * std4, mean4 + 2 * std4, n_samples, endpoint=True)

    data_plot_X, data_plot_Y, data_plot_Z, data_plot_W = np.meshgrid(xplot, yplot, zplot,
                                                                     wplot)  # Transform vectors to matrices through repetition

    data_plot_X_vectorized = data_plot_X.flatten()  # Vectorize the grid matrix data_plot_X
    data_plot_Y_vectorized = data_plot_Y.flatten()  # Vectorize the grid matrix data_plot_Y
    data_plot_Z_vectorized = data_plot_Z.flatten()  # Vectorize the grid matrix data_plot_Z
    data_plot_W_vectorized = data_plot_W.flatten()  # Vectorize the grid matrix data_plot_W

    # multivariate domain of pdf function
    data_plot = np.transpose(
        np.vstack((data_plot_X_vectorized, data_plot_Y_vectorized, data_plot_Z_vectorized, data_plot_W_vectorized)))
    #print("data plot shape=", data_plot.shape)
    #print('data plot= ', data_plot)
    # use kde to obtain pdf function fit to obtained domain
    bandwidth_kde = 0.5  # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth

    kernel_function = 'gaussian'
    kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(data_samples)
    # considered that kde return a log density
    kde_LogDensity_estimate = kde_object.score_samples(data_plot)
    kde_estimate = np.exp(kde_LogDensity_estimate)
    #print("KDE estimate =", kde_estimate)
    # visualize result only for bivariate case
    # f2 = plt.figure(2)
    # ax2 = plt.axes(projection='3d')
    # ax2.plot_trisurf(data_plot[:, 0], data_plot[:, 1], kde_estimate, linewidth=0, antialiased=False)
    # plt.show()
    return data_plot, kde_estimate


# Conditional probability function:
# given as input the data_plot matrix in which pdf multivariate values are mapped, kernel density multivariate pdf and
# row under test of test dataset, find and return value that correspond to conditional prob p(x|cj)
# ex. bivariate case value of pdf mapped to [x,y] realization
def conditional_prob(data_plot, row_under_test, kde_estimate):
    # find index of data_plot row where values correspond to row under test (row of test dataset)
    # ex. search [5.1 3.6 2.3 1.7] --> (row under test) in data_plot --> (matrix in which are mapped pdf values)
    # take index which satisfy data_plot[index] == [5.1 3.6 2.3 1.7] (almost equal not equal)
    # and compute kde_estimate[index]
    index = 0
    for a in range(0, len(data_plot)):
        # verify float matching between row under test (features of test dataset) and kde function value
        if np.allclose(data_plot[a], row_under_test, atol=0.18):  # match found
            index = a
            break
    return kde_estimate[index]


# naive bayes classifier function:
# given as input train dataset with class vector and test dataset, produce as output class vector for test dataset
def naive_bayes_classifier(train_dataset, train_class, test_dataset):
    # training phase1
    # compute pmf of training class label vector
    unique_class, class_pmf = discrete_pmf(train_class)
    #print('Pmf of training class label: ', class_pmf)
    #print('Shape training pmf of class label: ', class_pmf.shape)
    #print('Len training pmf of class label: ', len(class_pmf))

    # training phase2
    # divide training dataset into j (len pf pmf of training class label) sub dataset and
    # estimate its probability density function (instances are realization of continuous r.v.)
    # consider now the naive condition (each feat. is ind. so p(x|cj) is equal to product of each p(xi|cj)
    d1 = {}  # dictionary of pdf for feature 1
    d2 = {}  # dictionary of pdf for feature 2
    d3 = {}  # dictionary of pdf for feature 3
    d4 = {}  # dictionary of pdf for feature 4
    len_sub_dataset = int(len(train_dataset) / len(class_pmf))  # len of each sub dataset
    print('len sub dataset = ', len_sub_dataset)
    # split training dataset and compute pdf with gaussian kernel density estimator
    for i in range(1, len(class_pmf) + 1):
        # print in case you need control
        # print('i =', i, ' indici = ', (i - 1) * len_sub_dataset, '-', len_sub_dataset * i)
        # print('Sub training dataset ', i, ' :', train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :])
        # print('Sub training dataset ', i, ' len:',len(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :]))
        # print('Sub training dataset ', i, ' shape:',train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), :].shape)

        # for each feature (4) compute pdf with univariate kernel density estimator and save it in a dictionary
        d1["sub_data_plot{0}".format(i)], d1["sub_pdf{0}".format(i)] = pdf_univariate(
            train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 0])
        d2["sub_data_plot{0}".format(i)], d2["sub_pdf{0}".format(i)] = pdf_univariate(
            train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 1])
        d3["sub_data_plot{0}".format(i)], d3["sub_pdf{0}".format(i)] = pdf_univariate(
            train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 2])
        d4["sub_data_plot{0}".format(i)], d4["sub_pdf{0}".format(i)] = pdf_univariate(
            train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 3])

    # test phase1
    # now compute the class vector for the test dataset by applying Bayes theorem
    # take each row of test dataset and assign it to the class value which maximize p(cj|x)...
    print('Computing class vector for test data set...')
    # per each row of test dataset find p(x|cj)*p(cj) product per each cj and then select the max
    c = []
    bayes_arg = []  # collect here the values p(x|cj)*p(cj) per each j (class)
    for k in range(0, len(test_dataset)):  # for each row of test dataset
        bayes_arg.clear()  # clear it at start of each row computation
        for j in range(1, len(class_pmf) + 1):  # compute p(x|cj)*p(cj) per each j (class)
            # (is important to note that in denominator of bayes theorem is not considered because its cost over j)
            # compute first the conditional prob. p(x|cj) by using conditional_prob_naive fun
            conditional_probability = conditional_prob_naive(d1, d2, d3, d4, test_dataset[k], j)
            #print('Coditional prob. value: ', conditional_probability)
            # collect into bayes_arg all j values p(x|cj)*p(cj)
            bayes_arg.append(conditional_probability * class_pmf[j - 1])
            #print('Bayes arg ciclo', j, ' : ', bayes_arg)
        # select the max take c value and append to new class vector
        max = np.argmax(bayes_arg)
        print('Max for row ', k, ' is: ', max)
        c.append(max)
    print('c2 returned from naive: ', c)
    print('c len: ', len(c))
    return c


# pdf univariate function:
# given as input np array whit only one column, compute and return domain on which pdf value are mapped and
# pdf estimated by univariate kernel method
def pdf_univariate(a):
    # compute len of samples and stats
    vector_samples_len = len(a)
    samples_std = np.std(a)
    # compute the max and min values of all samples and 2*sigma
    samples_max = max(a)
    samples_min = min(a)
    margin = 2 * samples_std

    #  compute the optimal bandwidth suggested by kernel method
    optimal_bandwidth = 1.06 * samples_std * np.power(vector_samples_len, -1 / 5)
    #  gen kernel density object
    kde_object = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth).fit(a.reshape(-1, 1))
    #  compute domain of estimated gaussian pdf
    X_plot = np.linspace(samples_min - margin, samples_max + margin, vector_samples_len)[:, np.newaxis]
    #  score_samples give as result the natural log density of estimated pdf
    kde_LogDensity_estimate = kde_object.score_samples(X_plot)
    #  elevate e to log to obtain the pdf
    kde_estimate = np.exp(kde_LogDensity_estimate)

    return X_plot, kde_estimate


# conditional_prob_naive fun:
# compute and return the value of conditional probability p(x|cj), given as input pdf functions and its domains for each
# feature (4) and j that identify sub class label. In this naive condition p(x|cj) is quel to product of
# p(x1|c1)*...*p(xn|cj) if there are n features.
def conditional_prob_naive(d1, d2, d3, d4, row_under_test, j):
    data_plot1, data_plot2, data_plot3, data_plot4 = d1["sub_data_plot{0}".format(j)], d2["sub_data_plot{0}".format(j)], \
                                                     d3["sub_data_plot{0}".format(j)], d4["sub_data_plot{0}".format(j)]
    # take values estimated by kde (per each feature)
    kde1, kde2, kde3, kde4 = d1["sub_pdf{0}".format(j)], d2["sub_pdf{0}".format(j)], \
                             d3["sub_pdf{0}".format(j)], d4["sub_pdf{0}".format(j)]

    index1, index2, index3, index4 = 0, 0, 0, 0

    # look for value of first feature and retrieve index that satisfy condition row_under_test[0] = data_plot1[index1]
    # almost equal not precisely equal because we are working with floats
    for a in range(0, len(data_plot1)):
        # verify float matching between row under test (features of test dataset) and kde function value
        if np.allclose(data_plot1[a], row_under_test[0], atol=0.18):  # match found
            #print('cond prob fun, index: ', a, 'val: ', data_plot1[a])
            #print('row under test: ', row_under_test[0])
            index1 = a
            break
    # look for value of second feature and retrieve index that satisfy condition row_under_test[1] = data_plot2[index2]
    # almost equal not precisely equal because we are working with floats
    for b in range(0, len(data_plot2)):
        # verify float matching between row under test (features of test dataset) and kde function value
        if np.allclose(data_plot2[b], row_under_test[1], atol=0.18):  # match found
            #print('cond prob fun, index: ', b, 'val: ', data_plot2[b])
            #print('row under test: ', row_under_test[1])
            index2 = b
            break
    # look for value of third feature and retrieve index that satisfy condition row_under_test[2] = data_plot3[index3]
    # almost equal not precisely equal because we are working with floats
    for c in range(0, len(data_plot3)):
        # verify float matching between row under test (features of test dataset) and kde function value
        if np.allclose(data_plot3[c], row_under_test[2], atol=0.18):  # match found
            #print('cond prob fun, index: ', c, 'val: ', data_plot3[c])
            #print('row under test: ', row_under_test[2])
            index3 = c
            break
    # look for value of fourth feature and retrieve index that satisfy condition row_under_test[3] = data_plot4[index4]
    # almost equal not precisely equal because we are working with floats
    for d in range(0, len(data_plot4)):
        # verify float matching between row under test (features of test dataset) and kde function value
        if np.allclose(data_plot4[d], row_under_test[3], atol=0.18):  # match found
            #print('cond prob fun, index: ', d, 'val: ', data_plot4[d])
            #print('row under test: ', row_under_test[3])
            index4 = d
            break
    # return now product of all pdf function that made up p(x|cj)
    return kde1[index1] * kde2[index2] * kde3[index3] * kde4[index4]


def gaussian_naive_bayes_classifier(train_dataset, train_class, test_dataset):
    # training phase1
    # compute pmf of training class label vector
    unique_class, class_pmf = discrete_pmf(train_class)
    #print('Pmf of training class label: ', class_pmf)
    #print('Shape training pmf of class label: ', class_pmf.shape)
    #print('Len training pmf of class label: ', len(class_pmf))

    # training phase2 now is not necessary to estimate p(x|cj) through kernel density estimator  because density
    # function of each feature stat is well known (Gaussian with given mean and std), split training dataset in j sub
    # dataset and estimate mean and variance that is what differentiates the distributions between features
    # and classes (for each class each feature has a different mean and variance) for feature 1 that belongs to class 1
    # we have a given mean and std and soo on
    len_sub_dataset = int(len(train_dataset) / len(class_pmf))  # len of each sub dataset
    d1 = {}  # dictionary of mean
    d2 = {}  # dictionary of std
    # split training dataset and compute pdf with gaussian kernel density estimator
    for i in range(1, len(class_pmf) + 1):
        # for each column compute its stat values (mean and std)
        d1["stat1{0}".format(i)] = np.mean(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 0])
        d1["stat2{0}".format(i)] = np.mean(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 1])
        d1["stat3{0}".format(i)] = np.mean(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 2])
        d1["stat4{0}".format(i)] = np.mean(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 3])
        d2["stat1{0}".format(i)] = np.std(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 0])
        d2["stat2{0}".format(i)] = np.std(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 1])
        d2["stat3{0}".format(i)] = np.std(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 2])
        d2["stat4{0}".format(i)] = np.std(train_dataset[(i - 1) * len_sub_dataset:(len_sub_dataset * i), 3])

    # test phase1
    # now compute the class vector for the test dataset by applying Bayes theorem
    # test each row of test dataset and assign them to class value which maximize p(cj|x)...
    print('Computing class vector for test data set...')
    # per each row of test dataset find p(x|cj)*p(cj) product per each cj and then select the max
    c = []
    bayes_arg = []  # values p(x|cj)*p(cj)
    for k in range(0, len(test_dataset)):  # for each row of test dataset
        bayes_arg.clear()  # clear it at start of each row computation
        for j in range(1, len(class_pmf) + 1):  # compute p(x|cj)*p(cj) per each j
            # (is important to note that in denominator of bayes theorem is not considered because its cost over j)
            # compute first the conditional prob. p(x|cj) by using directly gaussian and proper mean and std
            conditional_probability = conditional_gaussian(d1, d2, test_dataset[k], j)
            #print('Coditional prob. value: ', conditional_probability)
            # collect into bayes_arg all j values p(x|cj)*p(cj)
            bayes_arg.append(conditional_probability * class_pmf[j - 1])
            #print('Bayes arg ciclo', j, ' : ', bayes_arg)
        # select the max take c value and append to new class vector
        max = np.argmax(bayes_arg)
        print('Max for row ', k, ' is: ', max)
        c.append(max)
    print('c3 returned from gaussian naive: ', c)
    print('c len: ', len(c))
    return c


#  definition of gaussian function:
#  given as input a point x, mean value and std produce as output val of gaussian in x
def univariate_gaussian(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))

# conditional_gaussian fun:
# given as input dictionary 1 containing mean values of 1,2,3,4 features of test dataset,
# dictionary 2 containing std values of 1,2,3,4 features of test dataset, row under test of test dataset,
# and j (index of class), produce as output conditional probability p(x|cj) considering that each feature is
# independent from each others
def conditional_gaussian(d1, d2, row_under_test, j):
    # for each column (4) compute pdf value with given mean and variance and then make product of that
    conditional_probability1 = univariate_gaussian(row_under_test[0], d1["stat1{0}".format(j)],
                                                   d2["stat1{0}".format(j)])  # column1
    conditional_probability2 = univariate_gaussian(row_under_test[1], d1["stat2{0}".format(j)],
                                                   d2["stat2{0}".format(j)])  # column2
    conditional_probability3 = univariate_gaussian(row_under_test[2], d1["stat3{0}".format(j)],
                                                   d2["stat3{0}".format(j)])  # column3
    conditional_probability4 = univariate_gaussian(row_under_test[3], d1["stat4{0}".format(j)],
                                                   d2["stat4{0}".format(j)])  # column4
    return conditional_probability1 * conditional_probability2 * conditional_probability3 * conditional_probability4
