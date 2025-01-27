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

def train_and_predict_NNLocalGP(row_idx, train_x, train_y, test_x, test_y, K_num = 100, Device = 'cpu', PCA_trans = 'None'):

    input_point = test_y[row_idx,:]
    local_train_x, local_train_y = Tools.find_k_nearest_neighbors_CPU(input_point, train_x, train_y, k = K_num)
    bounds = bound.get_bounds(local_train_x)
    
    NNLocalGP_models, NNLocalGP_likelihoods = Training.train_one_row_NNLocalGP_Parallel(train_x, train_y, test_y, row_idx, 
                                                                                        FeatureE.FeatureExtractor_6, 
                                                                                        covar_type = 'RBF', k_num = 100, lr=0.05, 
                                                                                        num_iterations=5000, patience=10, device=Device)
    
    preds = Prediction.full_preds(NNLocalGP_models, NNLocalGP_likelihoods, test_x[row_idx,:].unsqueeze(0).to(Device)).cpu().detach().numpy()
    if PCA_trans != 'None':
        preds = PCA_trans.inverse_transform(preds)

    estimated_params, func_loss = Estimation.multi_start_estimation(NNLocalGP_models, NNLocalGP_likelihoods, row_idx, test_y, bounds, Estimation.estimate_params_Adam, 
                                                                    num_starts=5, num_iterations=2000, lr=0.01, patience=20, 
                                                                    attraction_threshold=0.2, repulsion_strength=0.1, device=Device)

    return preds, estimated_params, func_loss




import os

output_file = 'Result/NNLocalGP_preds_MSE_NInd.csv'


if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('Iteration,MSE_NNLocalGP,MSE_estimated_NNLocalGP,func_loss\n')



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

    results = Parallel(n_jobs=-1)(delayed(train_and_predict_NNLocalGP)(row_idx, train_x, train_y, test_x, test_y) for row_idx in range(test_y.shape[0]))

    full_test_preds_NNLocalGP = [item[0] for item in results]
    full_test_estimated_params_NNLocalGP = [item[1] for item in results]
    full_test_estimated_params_NNLocalGP_loss = [item[2] for item in results]

    full_test_preds_NNLocalGP = np.array(full_test_preds_NNLocalGP)
    full_test_estimated_params_NNLocalGP = np.array(full_test_estimated_params_NNLocalGP)

    MSE_NNLocalGP = np.mean((full_test_preds_NNLocalGP - test_y.numpy()) ** 2)
    MSE_estimated_NNLocalGP = np.mean((full_test_estimated_params_NNLocalGP - test_x.numpy()) ** 2)
    Mean_loss = np.mean(full_test_estimated_params_NNLocalGP_loss)

    with open(output_file, 'a') as f:
        f.write(f"{i + 1},{MSE_NNLocalGP},{MSE_estimated_NNLocalGP},{Mean_loss}\n")



# nohup python NNLocalGP_NInd.py > NNLocalGP_NIndout.log 2>&1 &