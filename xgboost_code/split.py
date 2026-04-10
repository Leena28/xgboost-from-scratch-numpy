from loss import gradients,hessians,x 
import numpy as np

l2=1.0
gamma=0.0

def compute_leaf_value(gradients,hessians):
    
    w=-np.sum(gradients)/(np.sum(hessians)+l2)
    return w


#w=compute_leaf_value(gradients,hessians)
#print(w)

def compute_gain(G_left, H_left, G_right, H_right, G, H):

    left_score=(G_left**2)/(H_left+l2)
    right_score=(G_right**2)/(H_right+l2)
    parent=(G**2)/(H+l2)
    gain=left_score+right_score-parent-gamma

    return gain

def find_exact_greedy_split(x,gradients,hessians):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    best_default_direction = None

    num_features = x.shape[1]
    
    for i in range(num_features):
        feature = x[:, i]

        # Sparsity mask
        missing_mask = feature == 0
        non_missing_mask = ~missing_mask

        # Separate missing
        feature_non_missing = feature[non_missing_mask]
        gradients_non_missing = gradients[non_missing_mask]
        hessians_non_missing = hessians[non_missing_mask]

        # Missing stats
        G_missing = np.sum(gradients[missing_mask])
        H_missing = np.sum(hessians[missing_mask])

        # Sort non-missing
        sorted_idx = np.argsort(feature_non_missing)
        sorted_feature = feature_non_missing[sorted_idx]
        sorted_gradients = gradients_non_missing[sorted_idx]
        sorted_hessians = hessians_non_missing[sorted_idx]

        G_total = np.sum(sorted_gradients) + G_missing
        H_total = np.sum(sorted_hessians) + H_missing

        for j in range(len(sorted_feature) - 1):

            split_point = (sorted_feature[j] + sorted_feature[j+1]) / 2

            left = sorted_feature <= split_point
            right = sorted_feature > split_point

            G_left = np.sum(sorted_gradients[left])
            H_left = np.sum(sorted_hessians[left])

            G_right = np.sum(sorted_gradients[right])
            H_right = np.sum(sorted_hessians[right])

            # Case 1 Missing Left
            gain_left = compute_gain(G_left + G_missing,H_left + H_missing,G_right,H_right,G_total,H_total)

            # Case 2 Missing Right
            gain_right = compute_gain(G_left,H_left,G_right + G_missing,H_right + H_missing,G_total,H_total)

            # Choose best direction
            if gain_left > gain_right:
                gain = gain_left
                default_direction = "left"
            else:
                gain = gain_right
                default_direction = "right"

            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_threshold = split_point
                best_default_direction = default_direction

    return best_gain, best_feature, best_threshold, best_default_direction

def find_approximate_split(x,gradients,hessians):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_default_direction = None

        num_features = x.shape[1]

        sketch_eps = [0.25, 0.5, 0.75]

        for i in range(num_features):

            feature = x[:, i]

            # Sparsity mask
            missing_mask = feature == 0
            non_missing_mask = ~missing_mask

            # Separate missing
            feature_non_missing = feature[non_missing_mask]
            gradients_non_missing = gradients[non_missing_mask]
            hessians_non_missing = hessians[non_missing_mask]

            # Missing stats
            G_missing = np.sum(gradients[missing_mask])
            H_missing = np.sum(hessians[missing_mask])

            # Sort non-missing
            sorted_idx = np.argsort(feature_non_missing)

            sorted_feature = feature_non_missing[sorted_idx]
            sorted_gradients = gradients_non_missing[sorted_idx]
            sorted_hessians = hessians_non_missing[sorted_idx]

            # Total stats
            G_total = np.sum(sorted_gradients) + G_missing
            H_total = np.sum(sorted_hessians) + H_missing

            # Cumulative hessian (weighted quantile)
            cumulative_hessian = np.cumsum(sorted_hessians)

            # Quantile boundaries
            boundaries = np.array(sketch_eps) * cumulative_hessian[-1]

            # Candidate splits
            for boundary in boundaries:

                idx = np.searchsorted(cumulative_hessian, boundary)

                if idx >= len(sorted_feature):
                    continue

                threshold = sorted_feature[idx]

                left_indices = sorted_feature <= threshold
                right_indices = sorted_feature > threshold

                G_left = np.sum(sorted_gradients[left_indices])
                H_left = np.sum(sorted_hessians[left_indices])

                G_right = np.sum(sorted_gradients[right_indices])
                H_right = np.sum(sorted_hessians[right_indices])

                # Case 1- Missing Left
                gain_left = compute_gain(G_left + G_missing,H_left + H_missing,G_right,H_right,G_total,H_total)
                
                # Case 2- Missing Right
                gain_right = compute_gain(G_left,H_left,G_right + G_missing,H_right + H_missing,G_total,H_total)

                # Choose best direction
                if gain_left > gain_right:
                    gain = gain_left
                    default_direction = "left"
                else:
                    gain = gain_right
                    default_direction = "right"

                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
                    best_default_direction = default_direction

        return best_gain, best_feature, best_threshold, best_default_direction


def find_best_split(X, gradients, hessians, method='exact'):
            
        if method=='exact':
            best_split=find_exact_greedy_split(X, gradients, hessians)
        elif method=='approximate':
             best_split=find_approximate_split(X, gradients, hessians)
        return best_split     

    
    

               