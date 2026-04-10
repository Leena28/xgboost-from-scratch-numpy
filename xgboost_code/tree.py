import numpy as np
from loss import compute_gradients_and_hessians
from split import find_best_split,compute_leaf_value


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, default_direction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.default_direction = default_direction


def build_tree(X, gradients, hessians, depth=0, max_depth=3):

    # Stopping condition
    if depth >= max_depth:
        leaf_value = compute_leaf_value(gradients, hessians)
        return Node(value=leaf_value)

    # Find best split
    best_gain, best_feature, best_threshold,default_direction= find_best_split(X, gradients, hessians,'exact')

    # If no gain then make leaf
    if best_gain <= 0 or best_feature is None:
        leaf_value = compute_leaf_value(gradients, hessians)
        return Node(value=leaf_value)

    # Split dataset
    left_indices = X[:, best_feature] <= best_threshold
    right_indices = X[:, best_feature] > best_threshold

    # Split data
    X_left = X[left_indices]
    X_right = X[right_indices]

    grad_left = gradients[left_indices]
    grad_right = gradients[right_indices]

    hess_left = hessians[left_indices]
    hess_right = hessians[right_indices]

    # Build left subtree
    left_node = build_tree(X_left,grad_left,hess_left,depth + 1,max_depth)

    # Build right subtree
    right_node = build_tree(X_right,grad_right,hess_right,depth + 1,max_depth)

    # Return node
    return Node(feature=best_feature,threshold=best_threshold,left=left_node,right=right_node,default_direction=default_direction)