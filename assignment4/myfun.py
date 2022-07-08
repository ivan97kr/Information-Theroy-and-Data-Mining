import math
import numpy as np


#   entropy fun:
#   computes the entropy of a discrete random variable given its probability mass function vector p = [p1; p2; :::; pN].
def entropy(p):
    # lunghezza del vettore massa di probabilità
    n = len(p)
    # var per il calcolo dell'entropia
    h = 0
    # sommatoria che costituisce la entropy function
    for i in range(0, n):
        if p[i] == 0:  # avoid lof of 0
            continue
        else:
            h = h + p[i] * math.log((1 / p[i]), 2)
    return h


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


#   multivariate pmf function:
#   function to estimate multivariate pmf, given as input a matrix with n_rows = instances
#   of data_matrix and n_column = features of each instance (2 in this exercise case), return vector of unique values
#   and square matrix of estimated joint probability mass function
def pmf_multivariate(data_matrix):
    #  number of instances and features
    rows, columns = data_matrix.shape

    # use unique np function with axis=0 to return unique instances (rows) values ordered and then the count of each
    # row repetition
    unique_rows_ordered, count_rep_vector = np.unique(data_matrix, axis=0, return_counts=True)
    # by definition the pmf multivariate is the number of repetition of each row divided by the total number of rows
    pmf_vector = count_rep_vector / rows
    print('Joint pmf_vector len is: ', len(pmf_vector))
    print('Joint pmf_vector shape: ', pmf_vector.shape)
    print('Joint pmf_vector type: ', type(pmf_vector))

    # in order to obtain a square matrix from pmf_vector computed now
    # (to perfectly work with mutual information fun): put pmf_vector in a square matrix
    # first find max and min values for each dimension to correct set up dimension the pmf bidimensional matrix
    xmax, ymax = unique_rows_ordered.max(axis=0)
    xmin, ymin = unique_rows_ordered.min(axis=0)
    # compute the len of each dimension
    xlen = xmax - xmin
    ylen = ymax - ymin
    # take the max dimension in order to create a square matrix
    dim = max(xlen, ylen)
    # initialize a square matrix to zeros value
    pmf_matrix = np.zeros((dim + 1, dim + 1))
    # fill this matrix with the values of pmf_vector
    for i in range(0, len(unique_rows_ordered)):
        # take the index of each pmf vector sample in order to normalize the domain from (43,...,79) to
        # (0,...,xmax) same for y component in order to obtain a smaller matrix
        index = unique_rows_ordered[i] - [xmin, ymin]
        # ex: first pmf_vector value is the prob. of couple [x = 43, y = 30], so take these indexes and subtract them
        # of [xmin, ymin] obtaining [43-43, 30-20]. So in position [0, 10] insert the value of probability of that
        # event

        # insert the probability value into matrix in the correct position
        pmf_matrix[index[0], index[1]] = pmf_vector[i]
    return unique_rows_ordered, pmf_matrix


def pmf_multivariate2(data_matrix):
    # returns the number of rows and columns of the data_matrix
    rows, columns = data_matrix.shape
    # the parameter axis=0 allows to count the unique rows
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)

    return unique_rows_array, pmf_vector / rows  # To obtain the probability, the count must be normalized to the
    # total count of samples


#   mutual information fun1:
#   computes the mutual information of two generic discrete random variables given their joint and marginal p.m.f.
def mutual_information1(pxy, px, py):
    #  numero delle righe (possibili eventi v.a. x)
    nx = len(px)
    #  numero delle colonne (possibili eventi v.a. y)
    ny = len(py)
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            if pxy[i][j] == 0 or px[i] == 0 or py[j] == 0:  # avoid log of 0
                continue
            else:
                h = h + (pxy[i][j] * math.log(((pxy[i][j]) / (px[i] * py[j])), 2))
    return h


#   mutual information fun2:
#   computes the mutual information of two generic discrete random variables given only their joint p.m.f.
def mutual_information2(pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = pxy.shape[0]
    #  numero delle colonne (possibili eventi v.a. y)
    ny = pxy.shape[1]
    #   calcolo la probabilità marginale di x
    py = pxy.sum(axis=0)  # 0 somma tutti gli elementi di ogni colonna, restituisce vettore riga
    #   calcolo la probabilità marginale di y
    px = pxy.sum(axis=1)  # 1 somma tutti gli elementi di ogni riga, restituisce vettore colonna
    # var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la conditional entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            if pxy[i][j] == 0 or px[i] == 0 or py[j] == 0:  # avoid divide by 0
                continue
            else:
                h = h + (pxy[i][j] * math.log(((pxy[i][j]) / (px[i] * py[j])), 2))
    return h


#   joint entropy fun:
#   compute the joint entropy function gien as input the joint probability mass function
def joint_entropy(pxy):
    #  numero delle righe (possibili eventi v.a. x)
    nx = pxy.shape[0]  # 0 conta le righe, 1 conta le colonne
    #  numero delle colonne (possibili eventi v.a. y)
    ny = pxy.shape[1]
    #  var per il calcolo dell'entropia
    h = 0
    #  doppia sommatoria che costituisce la joint entropy function
    for i in range(0, nx):
        for j in range(0, ny):
            if pxy[i][j] == 0:  # avoid divide by 0
                continue
            else:
                h = h + pxy[i][j] * math.log((1 / pxy[i][j]), 2)
    return h
