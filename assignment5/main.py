from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import metrics
import myfun


iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target

print("Data shape =", data_matrix.shape)    #The Iris dataset is a 150x4 data matrix
print("Data matrix =", data_matrix)
print("Class vector =", class_vector)

# take 25 rows of each class of iris as training dataset and other 25 rows as test dataset
training_dataset = np.vstack((data_matrix[0:25, :], data_matrix[50:75, :], data_matrix[100:125, :]))
print('training dataset: ', training_dataset)
print('training dataset shape: ', training_dataset.shape)
print('training dataset len: ', len(training_dataset))
# take class vector
training_class_vector = np.hstack((class_vector[0:25], class_vector[50:75], class_vector[100:125]))
print('training class vector: ', training_class_vector)
print('training class vector shape: ', training_class_vector.shape)
print('training class vector len: ', len(training_class_vector))

test_dataset = np.vstack((data_matrix[25:50, :], data_matrix[75:100, :], data_matrix[125:150, :]))
print('test dataset: ', test_dataset)
print('test dataset shape: ', test_dataset.shape)
print('test dataset len: ', len(test_dataset))

# class vector for test dataset using full bayes classifier
c1 = myfun.bayes_classifier(training_dataset, training_class_vector, test_dataset)

# class vector for test dataset using naive bayes classifier
c2 = myfun.naive_bayes_classifier(training_dataset, training_class_vector, test_dataset)

# class vector for test dataset using naive bayes classifier considering that features are gaussian with given std,mean
c3 = myfun.gaussian_naive_bayes_classifier(training_dataset, training_class_vector, test_dataset)

print('Accuracy full bayes classifier: ', metrics.accuracy_score(training_class_vector, c1))
print('Accuracy naive bayes classifier: ', metrics.accuracy_score(training_class_vector, c2))
print('Accuracy gaussian naive bayes classifier: ', metrics.accuracy_score(training_class_vector, c3 ))