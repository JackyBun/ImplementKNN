from scipy.spatial import KDTree
import numpy as np

class KNN_KDTree:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.tree = KDTree(X)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            _, indices = self.tree.query(X[i].reshape(1, -1), k=self.k)
            labels = [self.y_train[idx] for idx in indices[0]]
            y_pred.append(max(set(labels), key=labels.count))
        return np.array(y_pred)
