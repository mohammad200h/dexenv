import numpy as np
import torch
from loguru import logger

from dexenv.utils.torch_utils import torch_float


def cal_gae(gamma, lam, rewards, value_estimates, last_value, dones, timeout=None):
    device = value_estimates.device
    rewards = torch_float(rewards, device)
    value_estimates = torch_float(value_estimates, device)
    last_value = torch_float(last_value, device)
    
    # Log NaN tracking for inputs
    if torch.isnan(rewards).any():
        logger.error(f"[cal_gae] Input rewards contains NaN! NaN count: {torch.isnan(rewards).sum()}, stats - min: {rewards.min()}, max: {rewards.max()}, mean: {rewards.mean()}")
    if torch.isnan(value_estimates).any():
        logger.error(f"[cal_gae] Input value_estimates contains NaN! NaN count: {torch.isnan(value_estimates).sum()}, stats - min: {value_estimates.min()}, max: {value_estimates.max()}, mean: {value_estimates.mean()}")
    if torch.isnan(last_value).any():
        logger.error(f"[cal_gae] Input last_value contains NaN! NaN count: {torch.isnan(last_value).sum()}, stats - min: {last_value.min()}, max: {last_value.max()}, mean: {last_value.mean()}")
    
    if len(value_estimates.shape) > 1:
        last_value = last_value.view(1, -1)
    dones = torch_float(dones, device)
    advs = torch.zeros_like(rewards, device=device)
    last_gae_lam = 0
    value_estimates = torch.cat((value_estimates,
                                 last_value),
                                dim=0)
    
    # Log NaN tracking after concatenation
    if torch.isnan(value_estimates).any():
        logger.error(f"[cal_gae] value_estimates after concat contains NaN! NaN count: {torch.isnan(value_estimates).sum()}")

    for t in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[t]
        if timeout is not None:
            non_terminal2 = torch.logical_xor(non_terminal, timeout[t])
        else:
            non_terminal2 = non_terminal
        delta = rewards[t] + gamma * value_estimates[t + 1] * non_terminal2 - value_estimates[t]
        
        # Log NaN tracking for delta
        if torch.isnan(delta).any():
            logger.error(f"[cal_gae] delta at t={t} contains NaN! delta stats - min: {delta.min()}, max: {delta.max()}, mean: {delta.mean()}")
            logger.error(f"[cal_gae] rewards[t] stats - min: {rewards[t].min()}, max: {rewards[t].max()}, mean: {rewards[t].mean()}")
            logger.error(f"[cal_gae] value_estimates[t+1] stats - min: {value_estimates[t+1].min()}, max: {value_estimates[t+1].max()}, mean: {value_estimates[t+1].mean()}")
            logger.error(f"[cal_gae] value_estimates[t] stats - min: {value_estimates[t].min()}, max: {value_estimates[t].max()}, mean: {value_estimates[t].mean()}")
            logger.error(f"[cal_gae] gamma: {gamma}, non_terminal2 stats - min: {non_terminal2.min()}, max: {non_terminal2.max()}, mean: {non_terminal2.mean()}")
        
        last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
        
        # Log NaN tracking for last_gae_lam
        if torch.isnan(last_gae_lam).any():
            logger.error(f"[cal_gae] last_gae_lam at t={t} contains NaN! last_gae_lam stats - min: {last_gae_lam.min()}, max: {last_gae_lam.max()}, mean: {last_gae_lam.mean()}")
            logger.error(f"[cal_gae] delta stats - min: {delta.min()}, max: {delta.max()}, mean: {delta.mean()}")
            logger.error(f"[cal_gae] gamma: {gamma}, lam: {lam}, non_terminal stats - min: {non_terminal.min()}, max: {non_terminal.max()}, mean: {non_terminal.mean()}")
        
        advs[t] = last_gae_lam.clone()
    
    # Log NaN tracking for final advs
    if torch.isnan(advs).any():
        logger.error(f"[cal_gae] Final advs contains NaN! NaN count: {torch.isnan(advs).sum()}, stats - min: {advs.min()}, max: {advs.max()}, mean: {advs.mean()}")
    
    return advs
