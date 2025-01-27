"""
File: Tools.py
Author: Hongjin Ren
Description: Some tools which can help analyze some data, I wish.

"""

#############################################################################
## Package imports
#############################################################################

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
from scipy.stats import qmc, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.sparse import random as sparse_random


#############################################################################
## 
#############################################################################


def Print_percentiles(mse_array):
    """
    Prints the 1st, 2nd, and 3rd quantiles of the given data.
    """
    return {
        '25th Perc.': np.percentile(mse_array, 25),
        'Median': np.percentile(mse_array, 50),
        '75th Perc.': np.percentile(mse_array, 75)
    }


#############################################################################
## Set up two function suit for different device
#############################################################################

def find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k):
    distances = [distance.euclidean(input_point, train_pt) for train_pt in train_y]
    nearest_neighbors = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    return train_x[nearest_neighbors], train_y[nearest_neighbors]

def find_k_nearest_neighbors_GPU(input_point, train_x, train_y, k):

    input_point = input_point.view(1, -1).expand_as(train_y)
    distances = torch.norm(input_point - train_y, dim=1)
    _, nearest_neighbor_idxs = torch.topk(distances, k, largest=False, sorted=True)
    nearest_train_x = train_x[nearest_neighbor_idxs]
    nearest_train_y = train_y[nearest_neighbor_idxs]
    
    return nearest_train_x, nearest_train_y




#############################################################################
## Set up two function suit for different device
#############################################################################


def find_k_nearest_neighbors_Mahalanobis(input_point, train_x, train_y, k):

    cov_matrix = np.cov(train_y, rowvar=False)
    inv_cov_matrix = inv(cov_matrix)
    
    def mahalanobis_dist(x, y):
        return mahalanobis(x, y, inv_cov_matrix)
    
    distances = [mahalanobis_dist(input_point, train_pt) for train_pt in train_y]
    
    nearest_neighbors_idx = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    
    return train_x[nearest_neighbors_idx], train_y[nearest_neighbors_idx]

#############################################################################
## 
#############################################################################

def select_subsequence(original_points, target_num_points):

    # Calculate the step to select points to approximately get the target number of points
    total_points = len(original_points)
    step = max(1, total_points // target_num_points)
    
    # Select points by stepping through the original sequence
    selected_points = original_points[::step]
    
    # Ensure we have exactly target_num_points by adjusting the selection if necessary
    if len(selected_points) > target_num_points:
        # If we selected too many points, trim the excess
        selected_points = selected_points[:target_num_points]
    elif len(selected_points) < target_num_points:
        # If we selected too few points, this indicates a rounding issue with step; handle as needed
        # This is a simple handling method and might need refinement based on specific requirements
        additional_indices = np.random.choice(range(total_points), size=target_num_points - len(selected_points), replace=False)
        additional_points = original_points[additional_indices]
        selected_points = np.vstack((selected_points, additional_points))
    
    return selected_points 





#############################################################################
## 
#############################################################################

def generate_MVN_datasets(dimension_x, dimension_y, inner_dim, num_train_locations, num_test_locations, seed_train, seed_test, variance, sparse_density = 0.3, relationship = 'Inde'):
    
    def kernel(X1, X2, length_scales, variance):
        length_scales = np.asarray(length_scales)
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * np.dot(X1_scaled, X2_scaled.T)
        return variance * np.exp(-0.5 * sqdist)

    # Generate Sobol sequences
    sobol_train_gen = qmc.Sobol(d=dimension_x, seed=seed_train)
    sobol_test_gen = qmc.Sobol(d=dimension_x, seed=seed_test)
    X_train = sobol_train_gen.random_base2(m=num_train_locations)
    X_test = sobol_test_gen.random_base2(m=num_test_locations)


    X_all = np.concatenate((X_train, X_test), axis=0)

    # Scale design locations
    l_bounds, u_bounds = [0.1] * dimension_x, [5] * dimension_x
    X_all = qmc.scale(X_all, l_bounds, u_bounds)

    Y_inner_all = np.zeros(((2**num_train_locations + 2**num_test_locations), inner_dim))

    for i in range(inner_dim):
        rng = np.random.default_rng(seed=i)
        length_scales_ard = rng.random(dimension_x) * 10 + 5
        # Calculate covariance matrices
        K_all = kernel(X_all, X_all, length_scales_ard, variance)


        # Define mean vectors and generate datasets
        # random_mean = round(np.random.uniform(-10, 10), 2)
        # random_mean = round(np.random.uniform(0, 20), 4)
        # mean_vector_all = np.zeros((2**num_train_locations + 2**num_test_locations)) + random_mean
        mean_vector_all = np.zeros((2**num_train_locations + 2**num_test_locations))

        Y_tmp = multivariate_normal.rvs(mean=mean_vector_all, cov=K_all)

        Y_inner_all[:, i] = Y_tmp


    rng_y = np.random.default_rng(seed=999)

    if relationship == 'Inde':
        matrix = np.diag(rng_y.uniform(1, 10, size=(dimension_y,)))
        Y_all = Y_inner_all @ matrix
    else:
        sparse_matrix = sparse_random(
            inner_dim,
            dimension_y,
            density=sparse_density,
            random_state=999,
            data_rvs=lambda size: rng_y.uniform(0, 1, size=size)
        )
        matrix = sparse_matrix.toarray()

        for col_idx in range(matrix.shape[1]):
            if not np.any(matrix[:, col_idx]):
                row_idx = np.random.randint(0, matrix.shape[0])
                matrix[row_idx, col_idx] = rng_y.uniform(0, 0.1)

        # matrix = rng_y.uniform(low=9, high=10, size=(inner_dim, dimension_y))
        # matrix = rng_y.random(size=(inner_dim, dimension_y))
        # A = rng_y.normal(size=(inner_dim, dimension_y))
        # positive_definite_A = A.T @ A
        # U, _, Vt = np.linalg.svd(positive_definite_A)
        # matrix = U[:, :dimension_y] @ Vt[:dimension_y, :]

        Y_all = Y_inner_all @ matrix


    tmp_mean_train = np.mean(Y_all, axis=0)
    tmp_std_train = np.std(Y_all, axis=0)
    Y_train_all_standardized = (Y_all - tmp_mean_train) / tmp_std_train


    X_train = X_all[:(2**num_train_locations), :]
    X_test = X_all[(2**num_train_locations + 5): -5, :]

    Y_train = Y_train_all_standardized[:(2**num_train_locations), :]
    Y_test = Y_train_all_standardized[(2**num_train_locations + 5): -5, :]
    
    return X_train, X_test, Y_train, Y_test



#############################################################################
## Save and Load
#############################################################################

def save_models_likelihoods(Models, Likelihoods, file_path):
    state_dicts = {
        'models': [model.state_dict() for model in Models],
        'likelihoods': [likelihood.state_dict() for likelihood in Likelihoods]
    }
    torch.save(state_dicts, file_path)


def load_models_likelihoods(file_path, model_class, likelihood_class, train_x, inducing_points, covar_type='RBF', device='cpu'):
    state_dicts = torch.load(file_path)
    
    Models = []
    Likelihoods = []
    for model_state, likelihood_state in zip(state_dicts['models'], state_dicts['likelihoods']):
        model = model_class(train_x, inducing_points=inducing_points, covar_type=covar_type)
        model.load_state_dict(model_state)
        model = model.to(device)
        
        likelihood = likelihood_class()
        likelihood.load_state_dict(likelihood_state)
        likelihood = likelihood.to(device)
        
        Models.append(model)
        Likelihoods.append(likelihood)
    
    return Models, Likelihoods

#################
##
################

def get_outlier_indices_iqr(data, outbound = 1.5):
    mask = np.ones(data.shape[0], dtype=bool)
    
    for i in range(data.shape[1]):
        Q1 = np.percentile(data[:, i], 25)
        Q3 = np.percentile(data[:, i], 75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - outbound * IQR
        upper_bound = Q3 + outbound * IQR
        
        mask = mask & (data[:, i] >= lower_bound) & (data[:, i] <= upper_bound)
    
    outlier_indices = np.where(~mask)[0]  
    return outlier_indices