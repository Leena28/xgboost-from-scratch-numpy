import numpy as np

from loss import compute_gradients_and_hessians
from tree import build_tree


class XGBoost:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=3,):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        base_score = None

    def fit(self, X, y):

        #y_pred = np.zeros(len(y))
        self.base_score = np.mean(y)
        y_pred = np.full(len(y), self.base_score)
        
        # Boosting loop
        for _ in range(self.n_estimators):

            gradients, hessians = compute_gradients_and_hessians(y, y_pred)

            tree = build_tree(X,gradients,hessians,depth=0,max_depth=self.max_depth)

            self.trees.append(tree)

            predictions = self.predict_tree(tree, X)

            y_pred += self.learning_rate * predictions

    def predict_tree(self, node, X):

        # Leaf node
        if node.value is not None:
            return np.full(X.shape[0], node.value)

        feature = node.feature
        threshold = node.threshold

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        predictions = np.zeros(X.shape[0])

        predictions[left_indices] = self.predict_tree(node.left,X[left_indices])

        predictions[right_indices] = self.predict_tree(node.right,X[right_indices])

        return predictions

    def predict(self, X):

        #y_pred = np.zeros(X.shape[0])
        y_pred = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            y_pred += self.learning_rate * self.predict_tree(tree, X)

        return y_pred