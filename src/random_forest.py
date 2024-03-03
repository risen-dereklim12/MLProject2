import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=100, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        for _ in range(n_trees):
            self.trees.append(DecisionTree(max_depth=max_depth))

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.mean(predictions, axis=1)