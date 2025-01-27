"""
File: Prediction.py
Author: Hongjin Ren
Description: Predict the reslut from Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import gpytorch



#############################################################################
## 
#############################################################################

def preds_for_one_model(model, likelihood, xxx):
    # Prediction of a column of the local data
    """
    Applicable Models: MultitaskGP, MultitaskVGP
    """
    
    model.eval()
    likelihood.eval()
    # with torch.no_grad(),gpytorch.settings.fast_pred_var():
    preds = likelihood(model(xxx)).mean
    return preds

def full_preds(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_one_model(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_one_model(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()


#############################################################################
## 
#############################################################################

def preds_for_SparseGP(model, likelihood, xxx):
    # Prediction of a column of the local data
    model.eval()
    likelihood.eval()
    # with torch.no_grad():
    preds = model.likelihood(model(xxx)).mean
    return preds

def full_preds_for_SparseGP(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_SparseGP(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_SparseGP(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()


#############################################################################
## 
#############################################################################


def preds_for_VGP(model, likelihood, xxx):
    # Prediction of a column of the local data
    model.eval()
    likelihood.eval()
    # with torch.no_grad():
    preds = model(xxx).mean
    return preds

def full_preds_for_VGP(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_VGP(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_VGP(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()

#############################################################################
## 
#############################################################################

def preds_for_column_var(model, likelihood, local_train_x):
    # Prediction of a column of the local data
    model.eval()
    likelihood.eval()
    preds_var = model(local_train_x).variance
    return preds_var

def full_preds_var(models, likelihoods, local_train_x):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_column_var(models[0], likelihoods[0], local_train_x).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_column_var(models[i], likelihoods[i],local_train_x).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()

#############################################################################
## 
#############################################################################
def preds_for_DNN(model, xxx):
    # Prediction of a column of the local data
    model.eval()  
    # with torch.no_grad():  
    preds = model(xxx)
    return preds


