import torch
import torch.nn as nn
from collections.abc import Sequence
from loguru import logger
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class DiagGaussianPolicy(nn.Module):
    def __init__(self,
                 body_net,
                 action_dim,
                 init_log_std=-0.2,
                 std_cond_in=False,
                 tanh_on_dist=False,
                 in_features=None,
                 clamp_log_std=False):  # add tanh on the action distribution
        super().__init__()
        self.std_cond_in = std_cond_in
        self.tanh_on_dist = tanh_on_dist
        self.body = body_net
        self.clamp_log_std = clamp_log_std

        if in_features is None:
            for i in reversed(range(len(self.body.fcs))):
                layer = self.body.fcs[i]
                if hasattr(layer, 'out_features'):
                    in_features = layer.out_features
                    break

        self.head_mean = nn.Linear(in_features, action_dim)
        if self.std_cond_in:
            self.head_logstd = nn.Linear(in_features, action_dim)
        else:
            self.head_logstd = nn.Parameter(torch.full((action_dim,),
                                                       init_log_std))

    def forward(self, x=None, body_x=None, **kwargs):
        if x is None and body_x is None:
            raise ValueError('One of [x, body_x] should be provided!')
        
        # Log input x for NaN tracking
        if x is not None:
            if isinstance(x, dict):
                for key, val in x.items():
                    if torch.is_tensor(val) and torch.isnan(val).any():
                        logger.error(f"[DiagGaussianPolicy] Input x dict['{key}'] contains NaN! shape: {val.shape}, NaN count: {torch.isnan(val).sum()}, NaN locations: {torch.isnan(val).nonzero()}")
                        logger.error(f"[DiagGaussianPolicy] Input x dict['{key}'] stats - min: {val.min()}, max: {val.max()}, mean: {val.mean()}")
            elif torch.is_tensor(x):
                if torch.isnan(x).any():
                    logger.error(f"[DiagGaussianPolicy] Input x contains NaN! x shape: {x.shape}, NaN count: {torch.isnan(x).sum()}, NaN locations: {torch.isnan(x).nonzero()}")
                    logger.error(f"[DiagGaussianPolicy] Input x stats - min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
        
        if body_x is None:
            body_x = self.body(x, **kwargs)
        
        body_out = body_x[0] if isinstance(body_x, Sequence) else body_x
        
        # Log body output for NaN tracking
        if torch.isnan(body_out).any():
            logger.error(f"[DiagGaussianPolicy] Body output contains NaN! body_out shape: {body_out.shape}, NaN count: {torch.isnan(body_out).sum()}, NaN locations: {torch.isnan(body_out).nonzero()}")
            logger.error(f"[DiagGaussianPolicy] Body output stats - min: {body_out.min()}, max: {body_out.max()}, mean: {body_out.mean()}")
            # Check body network weights for NaN
            if hasattr(self.body, 'named_parameters'):
                for name, param in self.body.named_parameters():
                    if torch.isnan(param).any():
                        logger.error(f"[DiagGaussianPolicy] Body network parameter '{name}' contains NaN! shape: {param.shape}, NaN count: {torch.isnan(param).sum()}")
                        logger.error(f"[DiagGaussianPolicy] Body network parameter '{name}' stats - min: {param.min()}, max: {param.max()}, mean: {param.mean()}")
            # Check body network modules (for Sequential networks)
            if hasattr(self.body, 'body') and isinstance(self.body.body, torch.nn.Sequential):
                for i, module in enumerate(self.body.body):
                    if isinstance(module, torch.nn.Linear):
                        if hasattr(module, 'weight') and torch.isnan(module.weight).any():
                            logger.error(f"[DiagGaussianPolicy] Body network layer {i} (Linear) weight contains NaN! shape: {module.weight.shape}")
                        if hasattr(module, 'bias') and module.bias is not None and torch.isnan(module.bias).any():
                            logger.error(f"[DiagGaussianPolicy] Body network layer {i} (Linear) bias contains NaN! shape: {module.bias.shape}")
        
        mean = self.head_mean(body_out)
        
        # Log mean before creating distribution
        if torch.isnan(mean).any():
            logger.error(f"[DiagGaussianPolicy] Mean contains NaN! mean shape: {mean.shape}, NaN count: {torch.isnan(mean).sum()}, NaN locations: {torch.isnan(mean).nonzero()}")
            logger.error(f"[DiagGaussianPolicy] Mean stats - min: {mean.min()}, max: {mean.max()}, mean: {mean.mean()}")
            logger.error(f"[DiagGaussianPolicy] Body output that produced NaN mean - min: {body_out.min()}, max: {body_out.max()}, mean: {body_out.mean()}")
            # Check head_mean weights
            if hasattr(self.head_mean, 'weight'):
                has_nan_weight = torch.isnan(self.head_mean.weight).any()
                logger.error(f"[DiagGaussianPolicy] head_mean.weight contains NaN: {has_nan_weight}")
                if has_nan_weight:
                    logger.error(f"[DiagGaussianPolicy] head_mean.weight stats - min: {self.head_mean.weight.min()}, max: {self.head_mean.weight.max()}, mean: {self.head_mean.weight.mean()}")
                    logger.error(f"[DiagGaussianPolicy] head_mean.weight NaN count: {torch.isnan(self.head_mean.weight).sum()} out of {self.head_mean.weight.numel()}")
            if hasattr(self.head_mean, 'bias') and self.head_mean.bias is not None:
                has_nan_bias = torch.isnan(self.head_mean.bias).any()
                logger.error(f"[DiagGaussianPolicy] head_mean.bias contains NaN: {has_nan_bias}")
                if has_nan_bias:
                    logger.error(f"[DiagGaussianPolicy] head_mean.bias stats - min: {self.head_mean.bias.min()}, max: {self.head_mean.bias.max()}, mean: {self.head_mean.bias.mean()}")
                    logger.error(f"[DiagGaussianPolicy] head_mean.bias NaN count: {torch.isnan(self.head_mean.bias).sum()} out of {self.head_mean.bias.numel()}")
        
        if self.std_cond_in:
            log_std = self.head_logstd(body_out)
        else:
            log_std = self.head_logstd.expand_as(mean)
        if self.clamp_log_std:
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Log std before creating distribution
        if torch.isnan(std).any():
            logger.error(f"[DiagGaussianPolicy] Std contains NaN! std shape: {std.shape}, NaN count: {torch.isnan(std).sum()}")
        
        action_dist = Independent(Normal(loc=mean, scale=std), 1)

        if self.tanh_on_dist:
            action_dist = TransformedDistribution(action_dist,
                                                  [TanhTransform(cache_size=1)])
        return action_dist, body_x
