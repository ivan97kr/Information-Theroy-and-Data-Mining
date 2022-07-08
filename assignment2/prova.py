import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.show()

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
