"""
File: Estimation.py
Author: Hongjin Ren
Description: Train the Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import numpy as np

import GP_functions.Prediction as Prediction
import tqdm

import math

#############################################################################
## 
#############################################################################


def estimate_params_for_one_model_Adam(model, likelihood, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cpu'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    model.eval()
    likelihood.eval()

    best_loss = float('inf')
    counter = 0

    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = torch.norm(likelihood.to(device)(model.to(device)(target_x)).mean - target_y, p=2).sum()
        loss.backward(retain_graph=True)
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()

        grad_norm = target_x.grad.data.norm(2).item()
        if grad_norm < attraction_threshold:
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss




def estimate_params_Adam(Models, Likelihoods, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cpu'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    best_loss = float('inf')
    counter = 0
    best_state = None
    
    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = torch.norm(Prediction.full_preds(Models, Likelihoods, target_x) - target_y, p=2).sum()
        loss.backward()
        optimizer.step()

        # Basinhopping of Attraction Law
        grad_norm = target_x.grad.data.norm(2).item()  #  log
        if grad_norm < attraction_threshold:
            # If the gradient norm is below a certain threshold, it may be stuck in a local minimum
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        # Parameter clipping, limiting the parameters to a specified range
        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        # iterator.set_postfix(loss=loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss






def estimate_params_Adam_VGP(Models, Likelihoods, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cpu'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    best_loss = float('inf')
    counter = 0
    best_state = None
    
    # iterator = tqdm.tqdm(range(num_iterations))
    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = torch.norm(Prediction.full_preds_for_VGP(Models, Likelihoods, target_x) - target_y, p=2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Basinhopping of Attraction Law
        grad_norm = target_x.grad.data.norm(2).item()
        if grad_norm < attraction_threshold:
            # If the gradient norm is below a certain threshold, it may be stuck in a local minimum
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        # Parameter clipping, limiting the parameters to a specified range
        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        # iterator.set_postfix(loss=loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss




def multi_start_estimation(model, likelihood, row_idx, test_y, param_ranges, estimate_function, num_starts=5, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device='cpu'):
    best_overall_loss = float('inf')
    best_overall_state = None

    quantiles = np.linspace(0.25, 0.75, num_starts)  
    
    for start in range(num_starts):
        # print(f"Starting optimization run {start+1}/{num_starts}")
        
        initial_guess = [np.quantile([min_val, max_val], quantiles[start]) for (min_val, max_val) in param_ranges]

        estimated_params, loss = estimate_function(
            model, likelihood, row_idx, test_y, initial_guess, param_ranges,
            num_iterations=num_iterations, lr=lr, patience=patience,
            attraction_threshold=attraction_threshold, repulsion_strength=repulsion_strength, device=device
        )

        if loss < best_overall_loss:
            best_overall_loss = loss
            best_overall_state = estimated_params

    return best_overall_state.detach().numpy(), best_overall_loss








def multi_start_estimation_nonliklihood(model, row_idx, test_y, param_ranges, estimate_function, num_starts=5, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device='cpu'):
    best_overall_loss = float('inf')
    best_overall_state = None

    quantiles = np.linspace(0.25, 0.75, num_starts)  
    
    for start in range(num_starts):
        # print(f"Starting optimization run {start+1}/{num_starts}")
        
        initial_guess = [np.quantile([min_val, max_val], quantiles[start]) for (min_val, max_val) in param_ranges]

        estimated_params, loss = estimate_function(
            model, row_idx, test_y, initial_guess, param_ranges,
            num_iterations=num_iterations, lr=lr, patience=patience,
            attraction_threshold=attraction_threshold, repulsion_strength=repulsion_strength, device=device
        )

        if loss < best_overall_loss:
            best_overall_loss = loss
            best_overall_state = estimated_params

    return best_overall_state.detach().numpy(), best_overall_loss







def estimate_params_for_DGP_Adam(DGP_model, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cuda'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    DGP_model.eval()

    best_loss = float('inf')
    counter = 0
    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = torch.norm(DGP_model.predict(target_x)[0] - target_y, p=2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()

        grad_norm = target_x.grad.data.norm(2).item()
        if grad_norm < attraction_threshold:
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        # iterator.set_postfix(loss=loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss






def estimate_params_for_NN_Adam(NN_model, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cuda'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    best_loss = float('inf')
    counter = 0
    best_state = None
    # iterator = tqdm.tqdm(range(num_iterations))

    # for i in iterator:
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = torch.norm(Prediction.preds_for_DNN(NN_model, target_x) - target_y, p=2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()

        grad_norm = target_x.grad.data.norm(2).item()
        if grad_norm < attraction_threshold:
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        # iterator.set_postfix(loss=loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss




