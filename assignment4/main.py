from sklearn import datasets
import myfun

#   load iris dataset from sklearn
iris = datasets.load_iris()
#   extract data matrix (n_instances * n_features = 150 * 4 ) and class vector (n_instances * 1 = 150 * 1) each
#   instances belongs to one of three class contained in the dataset (setosa, virginica, versicolor) represented by the
#   values (0,1,2) in the class vector and there are 50 instances per each class.
data_matrix, class_vector = iris.data, iris.target
#   print some data useful to verify
print("Data shape: ", data_matrix.shape)
print("Class vector: ", class_vector)
print("Class vector len: ", len(class_vector))

#  create dictionary to save each pmf of each feature
d = {}

#   for cycle to take all instances (150) per each features (4) one by one
for i in range(1, data_matrix.shape[1] + 1):

    # discretize samples of feature i by taking each value and multiply it by 10 and cast to int with astype
    samples_feature = (10 * data_matrix[:, i - 1:i]).astype(int)

    #   compute pmf of this feature (mono dimensional pmf)
    #   uso dictionary così a 'pmf{i}' associo il valore corrispondente
    unique_value_vector, d["pmf{0}".format(i)] = myfun.discrete_pmf(samples_feature)

#  useful data to verify each pmf
for i in range(1, 5):
    print('Pmf', i, ' len:  ', len(d["pmf{0}".format(i)]))
    print('Pmf', i, '  shape: ', d["pmf{0}".format(i)].shape)
    print('Pmf', i, '  sum: ', sum(d["pmf{0}".format(i)]))  # verify sum is 1

#   compute entropy of each features
entropy1 = myfun.entropy(d['pmf1'])
entropy2 = myfun.entropy(d['pmf2'])
entropy3 = myfun.entropy(d['pmf3'])
entropy4 = myfun.entropy(d['pmf4'])
print('Entropy of feature sepal length: ', entropy1)
print('Entropy of feature sepal width: ', entropy2)
print('Entropy of feature petal length: ', entropy3)
print('Entropy of feature petal length: ', entropy4)

# in order to compute mutual information between any pair of features:
# discretize and truncate data matrix in order to obtain only a subset of interested features
# take only sepal length and sepal width features in this case
discretized_data_matrix_sepal = (10 * data_matrix[:, [0, 1]]).astype(int)
print("Len of discretized data matrix sepal: ", len(discretized_data_matrix_sepal))
print('Shape of discretized data matrix: ', discretized_data_matrix_sepal.shape)

# estimate joint probability mass function by use pmf multivariate estimator function
unique_rows_sepal, pmf_multivariate_sepal = myfun.pmf_multivariate(discretized_data_matrix_sepal)

# print useful data to verify
print('Joint pmf matrix len: ', len(pmf_multivariate_sepal))
print('Joint pmf matrix shape: ', pmf_multivariate_sepal.shape)
print('Joint pmf matrix sum: ', sum(sum(pmf_multivariate_sepal)))  # verify sum 1

# obtain mutual information directly by using the estimated joint probability mass
mutual_info1 = myfun.mutual_information2(pmf_multivariate_sepal)
print('Mutual information: ', mutual_info1)

# is also possible to compute mutual info by using its relationship with joint entropy, compute it
print('Joint entropy of features couple: ', myfun.joint_entropy(pmf_multivariate_sepal))

# mutual_info = entropy_feature1 + entropy_feature2 - joint_entropy_1_2
mutual_info = entropy1 + entropy2 - myfun.joint_entropy(pmf_multivariate_sepal)
print('Mutual information by joint entropy: ', mutual_info)
