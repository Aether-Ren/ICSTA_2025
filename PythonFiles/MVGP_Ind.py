import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm


from joblib import Parallel, delayed


####

import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools
import GP_functions.FeatureE as FeatureE

####


dimension_x = 3
inner_dim = 6
dimension_y = 6

num_train_locations = 11
num_test_locations = 6

variance = 3.2

Device = 'cpu'

def train_and_predict_MVGP(MVGP_models, MVGP_likelihoods, row_idx, train_x, train_y, test_y, K_num = 100, Device = 'cpu'):

    input_point = test_y[row_idx,:]
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k = K_num)
    bounds = bound.get_bounds(local_train_x)
    

    estimated_params, func_loss = Estimation.multi_start_estimation(MVGP_models, MVGP_likelihoods, row_idx, test_y, bounds, Estimation.estimate_params_for_one_model_Adam, 
                                                                    num_starts=5, num_iterations=2000, lr=0.01, patience=10, 
                                                                    attraction_threshold=0.1, repulsion_strength=0.1, device=Device)

    return estimated_params




import os

output_file = 'Result/MVGP_preds_MSE_Ind.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,MSE_MVGP,MSE_estimated_MVGP\n')



for i in range(50):
    seed_train = i
    seed_test = i + 100

    X_train, X_test, Y_train, Y_test = Tools.generate_MVN_datasets(
        dimension_x, dimension_y, inner_dim, num_train_locations, num_test_locations,
        seed_train, seed_test, variance, sparse_density=0.3, relationship='Inde'
    )

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(Y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(Y_test, dtype=torch.float32)

    MVGP_models, MVGP_likelihoods = Training.train_full_MultitaskVGP(train_x, train_y, 
                                                                     covar_type = 'RBF', num_latents = inner_dim, num_inducing=100, 
                                                                     lr_hyper=0.05, lr_variational=0.1, num_iterations=5000, patience=20, device=Device)

    results = Parallel(n_jobs=-1)(delayed(train_and_predict_MVGP)(MVGP_models, MVGP_likelihoods, row_idx, train_x, train_y, test_y) for row_idx in range(test_y.shape[0]))

    full_test_preds_MVGP = Prediction.preds_for_one_model(MVGP_models, MVGP_likelihoods, test_x).detach().numpy()
    full_test_estimated_params_MVGP = np.array(np.vstack(results))

    MSE_MVGP = np.mean((full_test_preds_MVGP - test_y.numpy()) ** 2)
    MSE_estimated_MVGP = np.mean((full_test_estimated_params_MVGP - test_x.numpy()) ** 2)

    with open(output_file, 'a') as f:
        f.write(f"{i + 1},{MSE_MVGP},{MSE_estimated_MVGP}\n")



# nohup python MVGP_Ind.py > MVGP_Indout.log 2>&1 &