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
inner_dim = 3
dimension_y = 3

num_train_locations = 11
num_test_locations = 6

variance = 3.2

Device = 'cpu'

def train_and_predict_DGP(DGP_2, row_idx, train_x, train_y, test_y, K_num = 100, Device = 'cpu'):

    input_point = test_y[row_idx,:]
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k = K_num)
    bounds = bound.get_bounds(local_train_x)
    

    estimated_params, func_loss = Estimation.multi_start_estimation_nonliklihood(DGP_2, row_idx, test_y, bounds, Estimation.estimate_params_for_DGP_Adam, 
                                                                                 num_starts=5, num_iterations=2000, lr=0.01, patience=10, 
                                                                                 attraction_threshold=0.1, repulsion_strength=0.1, device=Device)

    return estimated_params, func_loss




import os

output_file = 'Result/DGP_preds_MSE_NInd.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,MSE_DGP,MSE_estimated_DGP,func_loss\n')



for i in range(50):
    seed_train = i
    seed_test = i + 100

    X_train, X_test, Y_train, Y_test = Tools.generate_MVN_datasets(
        dimension_x, dimension_y, inner_dim, num_train_locations, num_test_locations,
        seed_train, seed_test, variance, sparse_density=0.3, relationship='NInde'
    )

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(Y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(Y_test, dtype=torch.float32)

    DGP_2 = Training.train_full_DGP_2(train_x, train_y, num_hidden_dgp_dims = dimension_x, inducing_num = 100, num_iterations = 5000, patiences = 50, device=Device)

    results = Parallel(n_jobs=-1)(delayed(train_and_predict_DGP)(DGP_2, row_idx, train_x, train_y, test_y) for row_idx in range(test_y.shape[0]))

    full_test_mean = DGP_2.predict(test_x[0,:].unsqueeze(0))[0].detach().numpy()

    for row_idx in range(1,test_y.shape[0]):
        test_mean_tmp = DGP_2.predict(test_x[row_idx,:].unsqueeze(0))[0].detach().numpy()
        full_test_mean = np.vstack((full_test_mean, test_mean_tmp))

    full_test_preds_DGP = np.array(full_test_mean)
    # full_test_estimated_params_DGP = np.array(np.vstack(results))

    full_test_estimated_params_DGP = [item[0] for item in results]
    full_test_estimated_params_DGP_loss = [item[1] for item in results]

    full_test_estimated_params_DGP = np.array(full_test_estimated_params_DGP)

    MSE_DGP = np.mean((full_test_preds_DGP - test_y.numpy()) ** 2)
    MSE_estimated_DGP = np.mean((full_test_estimated_params_DGP - test_x.numpy()) ** 2)
    Mean_loss = np.mean(full_test_estimated_params_DGP_loss)

    with open(output_file, 'a') as f:
        f.write(f"{i + 1},{MSE_DGP},{MSE_estimated_DGP},{Mean_loss}\n")



# nohup python DGP_NInd.py > DGP_NIndout.log 2>&1 &