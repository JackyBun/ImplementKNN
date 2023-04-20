import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            distances = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            indices = np.argsort(distances)[:self.k]
            labels = [self.y_train[idx] for idx in indices]
            y_pred.append(max(set(labels), key=labels.count))
        return np.array(y_pred)
