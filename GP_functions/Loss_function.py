"""
File: Loss_function.py
Author: Hongjin Ren
Description: Create the loss function (Euclid, ...)

"""

#############################################################################
## Package imports
#############################################################################
import torch
import GP_functions.Prediction as Prediction


#############################################################################
## 
#############################################################################



def surrogate_loss_euclid(params, models, likelihoods, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred = Prediction.full_preds(models, likelihoods, params_tensor)
        loss = torch.norm(pred - test_y[row_idx,:]).pow(2).item()

    return loss



def surrogate_loss_for_one_model_euclid(params, model, likelihood, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred = Prediction.preds_for_one_model(model, likelihood, params_tensor)
        loss = torch.norm(pred - test_y[row_idx,:]).pow(2).item()

    return loss

def surrogate_loss_euclid_DNN(params, models, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred = Prediction.preds_for_DNN(models, params_tensor)
        loss = torch.norm(pred - test_y[row_idx,:]).pow(2).item()

    return loss



def surrogate_loss_euclid_DGP(params, model, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred_mean, full_test_var = model.predict(params_tensor)
        loss = torch.norm(pred_mean - test_y[row_idx,:]).pow(2).item()

    return loss


def surrogate_loss_euclid_VGP(params, models, likelihoods, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred = Prediction.full_preds_for_VGP(models, likelihoods, params_tensor)
        loss = torch.norm(pred - test_y[row_idx,:]).pow(2).item()

    return loss


#############################################################################
## Loss use to train NN
#############################################################################

def euclidean_distance_loss(output, target):
    return torch.norm(output - target).pow(2)
