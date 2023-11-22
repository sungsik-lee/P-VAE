import torch
import torch.nn as nn

def obs_compute_ELBOs(x, mu, logvar):
    batch_obs_ELBO = 0
    for i in range(len(x)):
        obs_ELBO =0.5 * torch.sum(mu[i]**2 + torch.exp(logvar[i]) - logvar[i] - 1)
        batch_obs_ELBO += obs_ELBO
    mean_obs_ELBO= batch_obs_ELBO/len(x)

    return mean_obs_ELBO

def obs_compute_recons(x, mask, y, data_dim, prob_num):
    mask = mask.squeeze()
    obs_data = x[mask]
    y= y.view(-1, data_dim, 2* prob_num)
    obs_pred = y[mask]
    obs_recon_loss = nn.CrossEntropyLoss(reduction = 'sum')(obs_pred, obs_data)

    obs_recon_loss = obs_recon_loss /len(x)

    return obs_recon_loss

def unobs_compute_ELBOs(x, mask, y, data_dim, prob_num):

    flipped_mask = ~mask
    flipped_mask = flipped_mask.squeeze()

    unobs_data = x[flipped_mask]

    y= y.view(-1, data_dim, 2* prob_num)
    unobs_pred = y[flipped_mask]

    unobs_recon_loss = nn.CrossEntropyLoss( reduction = 'sum')(unobs_pred, unobs_data)

    unobs_recon_loss = unobs_recon_loss/ len(x)

    return unobs_recon_loss