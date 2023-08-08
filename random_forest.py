import numpy as np
import pandas as pd

from random_forest import DecisionTree

class RandomForest():
    def __init__(self, n_estimators=100, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.decision_trees = None

    def generate_trees(self, x, y):
        indexes = np.random.randint(len(y), size=self.min_samples_split)
        return DecisionTree(x.iloc[indexes], y.iloc[indexes], self.min_samples_leaf)

    def fit(self, x, y):
        self.decision_trees = None
        self.decision_trees = [self.generate_trees(x, y) for i in range(self.n_estimators)]

    def predict(self, x):
        tree_predictions = [tree.predict(x) for tree in self.decision_trees]
        prediction = np.mean(tree_predictions, axis=0)
        return prediction
