import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

np.random.seed(0)
# import some data to play with
iris_X, iris_y = datasets.load_iris(return_X_y=True)

indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

from sklearn.externals import joblib

joblib.dump(knn, "knn.pkl")
