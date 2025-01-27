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

def train_and_predict_NN(NNmodel, row_idx, train_x, train_y, test_y, K_num = 100, Device = 'cpu'):

    input_point = test_y[row_idx,:]
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k = K_num)
    bounds = bound.get_bounds(local_train_x)
    

    estimated_params, func_loss = Estimation.multi_start_estimation_nonliklihood(NNmodel, row_idx, test_y, bounds, Estimation.estimate_params_for_NN_Adam, 
                                                                                 num_starts=5, num_iterations=2000, lr=0.01, patience=10, 
                                                                                 attraction_threshold=0.1, repulsion_strength=0.1, device=Device)

    return estimated_params




import os

output_file = 'Result/NN_preds_MSE_Ind.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,MSE_NN,MSE_estimated_NN\n')



for i in range(10,50):
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

    NNmodel = Training.train_DNN_4(train_x, train_y, num_iterations = 10000, device=Device)

    results = Parallel(n_jobs=-1)(delayed(train_and_predict_NN)(NNmodel, row_idx, train_x, train_y, test_y) for row_idx in range(test_y.shape[0]))


    full_test_preds_NN = Prediction.preds_for_DNN(NNmodel, test_x).detach().numpy()
    full_test_estimated_params_NN = np.array(np.vstack(results))

    MSE_NN = np.mean((full_test_preds_NN - test_y.numpy()) ** 2)
    MSE_estimated_NN = np.mean((full_test_estimated_params_NN - test_x.numpy()) ** 2)

    with open(output_file, 'a') as f:
        f.write(f"{i + 1},{MSE_NN},{MSE_estimated_NN}\n")



# nohup python NN_Ind.py > NN_Indout.log 2>&1 &