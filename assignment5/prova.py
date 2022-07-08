from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from mpl_toolkits import mplot3d  # This import registers the 3D projection, but is otherwise unused.

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target

print("Data shape =", data_matrix.shape)  # The Iris dataset is a 150x4 data matrix
print("Data matrix =", data_matrix)
print("Class vector =", class_vector)

########################################################################################################################
# List of n features-dimensional data points. Each row corresponds to a single data point
# We have selected two columns from the data matrix
# We may also select rows belonging to the same class
# data_samples = np.transpose(np.vstack((data_matrix[0:50, 0], data_matrix[0:50, 1])))   #1st class
# data_samples = np.transpose(np.vstack((data_matrix[50:100, 0], data_matrix[50:100, 1]))) #2nd class
# data_samples = np.transpose(np.vstack((data_matrix[100:150, 0], data_matrix[100:150, 1]))) #3rd class
data_samples = np.transpose(np.vstack((data_matrix[:, 0], data_matrix[:, 1])))  # All classes
print('Data samples= ', data_samples)

# print("Data samples=", data_samples)
# print("Data samples shape=", data_samples.shape)

feature_1_min, feature_1_max, feature_1_std, feature_1_mean = data_samples[:, 0].min(), data_samples[:,
                                                                                        0].max(), data_samples[:,
                                                                                                  0].std(), data_samples[
                                                                                                            :, 0].mean()
feature_2_min, feature_2_max, feature_2_std, feature_2_mean = data_samples[:, 1].min(), data_samples[:,
                                                                                        1].max(), data_samples[:,
                                                                                                  1].std(), data_samples[
                                                                                                            :, 1].mean()
# feature_3_min, feature_3_max, feature_3_std, feature_3_mean = data_samples[:, 2].min(), data_samples[:, 2].max(), data_samples[:, 2].std(), data_samples[:, 2].mean()

print("Feature 1 mean=", feature_1_mean)
print("Feature 2 mean=", feature_2_mean)
# print("Feature 3 mean=", feature_3_mean)

########################################################################################à
N_samples = 10
start_sample_1 = feature_1_mean - 2 * feature_1_std
start_sample_2 = feature_2_mean - 2 * feature_2_std
# start_sample_3 = feature_3_mean - 2*feature_3_std

stop_sample_1 = feature_1_mean + 2 * feature_1_std
stop_sample_2 = feature_2_mean + 2 * feature_2_std
# stop_sample_3 = feature_3_mean + 2*feature_3_std

X_plot = np.linspace(start_sample_1, stop_sample_1, N_samples, endpoint=True)  # row vector
Y_plot = np.linspace(start_sample_2, stop_sample_2, N_samples, endpoint=True)  # row vector
# Z_plot = np.linspace(start_sample_3, stop_sample_3, N_samples, endpoint=True)   #row vector


data_plot_X, data_plot_Y, = np.meshgrid(X_plot, Y_plot)  # Transform vectors to matrices through repetition
# Grids X,Y should be like: X=[[0, 1, 2,..., N], [0, 1, 2,..., N],...,[0, 1, 2,..., N]]
# and Y=[[0, 0, ..., 0], [1, 1, ...,1],...,[N, N,..., N]
print("data plot X=", data_plot_X)
data_plot_X_vectorized = data_plot_X.flatten()  # Vectorize the grid matrix data_plot_X
data_plot_Y_vectorized = data_plot_Y.flatten()  # Vectorize the grid matrix data_plot_Y
# data_plot_Z_vectorized = data_plot_Z.flatten()              #Vectorize the grid matrix data_plot_Y

# print("data plot X vectorized=", data_plot_X_vectorized)

data_plot = np.transpose(np.vstack((data_plot_X_vectorized, data_plot_Y_vectorized)))
print("data plot=", data_plot)
print('data plot shape=', data_plot.shape)

bandwidthKDE = 0.4  # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
kernelFunction = 'gaussian'  # Valid kernel functions are: ‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’
kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(data_samples)

kde_LogDensity_estimate = kde_object.score_samples(data_plot)
kde_estimate = np.exp(kde_LogDensity_estimate)
# print("KDE estimate =", kde_estimate)


# find index of data_plot value that correspond to p(x|cj)
index = 0
match = False
j = 0
# till match is not found
while not match:
    for a in range(0, len(data_plot)):
        # verify float matching
        if np.allclose(data_plot[a], [5.1, 3.5], atol=j):
            print('indice: ', a, 'val: ', data_plot[a])
            index = a
            match = True
            break
    # at the and o for if match is not found increment tolerance
    j = j + 0.1

print(kde_estimate[a])
