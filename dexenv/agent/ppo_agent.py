import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from loguru import logger
from omegaconf.dictconfig import DictConfig

from dexenv.utils.torch_utils import action_entropy
from dexenv.utils.torch_utils import action_from_dist
from dexenv.utils.torch_utils import action_log_prob
from dexenv.utils.torch_utils import clip_grad
from dexenv.utils.torch_utils import load_ckpt_data
from dexenv.utils.torch_utils import load_state_dict
from dexenv.utils.torch_utils import move_to
from dexenv.utils.torch_utils import save_model
from dexenv.utils.torch_utils import torch_float


@dataclass
class PPOAgent:
    actor: nn.Module
    critic: nn.Module
    cfg: DictConfig

    def __post_init__(self):
        move_to([self.actor, self.critic],
                device=self.cfg.alg.device)
        self.val_loss_criterion = nn.MSELoss().to(self.cfg.alg.device)

        self.optim_model = nn.ModuleList([self.actor, self.critic])

        optim_args = dict(
            lr=self.cfg.alg.policy_lr,
            amsgrad=True
        )

        optim_args['params'] = [{'params': self.actor.parameters(),
                                 'lr': self.cfg.alg.policy_lr},
                                {'params': self.critic.parameters(),
                                 'lr': self.cfg.alg.value_lr}
                                ]
        self.optimizer = optim.AdamW(**optim_args)

    @torch.no_grad()
    def get_action(self, ob, sample=True, get_action_only=False, *args, **kwargs):
        self.eval_mode()
        t_ob = torch_float(ob, device=self.cfg.alg.device)
        
        # Log NaN tracking for observations before passing to policy
        if isinstance(t_ob, dict):
            for key, val in t_ob.items():
                if torch.is_tensor(val) and torch.isnan(val).any():
                    logger.error(f"[PPOAgent.get_action] Observation dict['{key}'] contains NaN! shape: {val.shape}, NaN count: {torch.isnan(val).sum()}")
        elif torch.is_tensor(t_ob) and torch.isnan(t_ob).any():
            logger.error(f"[PPOAgent.get_action] Observation tensor contains NaN! shape: {t_ob.shape}, NaN count: {torch.isnan(t_ob).sum()}")
        
        act_dist, val = self.get_act_val(t_ob, no_val=get_action_only)
        action = action_from_dist(act_dist,
                                  sample=sample)

        if not get_action_only:
            log_prob = action_log_prob(action, act_dist)
            action_info = dict(
                log_prob=log_prob.detach(),
                val=val.detach()
            )
            entropy = action_entropy(act_dist, log_prob)
            action_info['entropy'] = entropy.detach()
        else:
            action_info = dict()
        return action.detach(), action_info

    def get_act_val(self, ob, no_val=False, *args, **kwargs):
        # Log NaN tracking for observations right before actor forward
        if isinstance(ob, dict):
            for key, val in ob.items():
                if torch.is_tensor(val) and torch.isnan(val).any():
                    logger.error(f"[PPOAgent.get_act_val] Observation dict['{key}'] contains NaN before actor forward! shape: {val.shape}, NaN count: {torch.isnan(val).sum()}")
        elif torch.is_tensor(ob) and torch.isnan(ob).any():
            logger.error(f"[PPOAgent.get_act_val] Observation tensor contains NaN before actor forward! shape: {ob.shape}, NaN count: {torch.isnan(ob).sum()}")
        
        act_dist, body_out = self.actor(ob)
        if no_val:
            val = None
        else:
            val, body_out = self.critic(x=ob)
            val = val.squeeze(-1)
        return act_dist, val

    @torch.no_grad()
    def get_val(self, ob, *args, **kwargs):
        self.eval_mode()
        ob = torch_float(ob, device=self.cfg.alg.device)
        val, body_out = self.critic(x=ob)
        val = val.squeeze(-1)
        return val

    def optimize(self, data, *args, **kwargs):
        pre_res = self.optim_preprocess(data)
        processed_data = pre_res

        self.optimizer.zero_grad(set_to_none=True)
        loss_res = self.cal_loss(**processed_data)
        loss, pg_loss, vf_loss, ratio, entropy, approx_kl, clip_frac = loss_res
        
        # Log NaN tracking for loss
        if torch.isnan(loss):
            logger.error(f"[PPOAgent.optimize] Loss is NaN! loss: {loss}, pg_loss: {pg_loss}, vf_loss: {vf_loss}")
        if torch.isnan(pg_loss):
            logger.error(f"[PPOAgent.optimize] pg_loss is NaN!")
        if torch.isnan(vf_loss):
            logger.error(f"[PPOAgent.optimize] vf_loss is NaN!")
        
        grad_norm = None
        loss.backward()
        
        # Log NaN tracking for gradients before clipping
        for name, param in self.actor.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' gradient contains NaN before clipping! NaN count: {torch.isnan(param.grad).sum()}")
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' gradient stats - min: {param.grad.min()}, max: {param.grad.max()}, mean: {param.grad.mean()}")
        
        grad_norm = clip_grad(self.optim_model.parameters(), self.cfg.alg.max_grad_norm)
        
        # Log NaN tracking for gradients after clipping
        for name, param in self.actor.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' gradient contains NaN after clipping! NaN count: {torch.isnan(param.grad).sum()}")
        
        # Log NaN tracking for weights before optimizer step
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' contains NaN BEFORE optimizer.step()! NaN count: {torch.isnan(param).sum()}")
        
        self.optimizer.step()
        
        # Log NaN tracking for weights after optimizer step
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' contains NaN AFTER optimizer.step()! NaN count: {torch.isnan(param).sum()}")
                logger.error(f"[PPOAgent.optimize] Actor parameter '{name}' stats - min: {param.min()}, max: {param.max()}, mean: {param.mean()}")
        optim_info = dict(
            pg_loss=pg_loss.item(),
            vf_loss=vf_loss.item(),
            total_loss=loss.item(),
            approx_kl=approx_kl.item(),
            clip_frac=clip_frac.item()
        )
        optim_info['entropy'] = entropy.item()
        if grad_norm is not None:
            optim_info['grad_norm'] = grad_norm
        return optim_info

    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=self.cfg.alg.device)
        ob = data['ob']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        # Log NaN tracking for input data
        if torch.isnan(ret).any():
            logger.error(f"[PPOAgent.optim_preprocess] ret contains NaN! NaN count: {torch.isnan(ret).sum()}, stats - min: {ret.min()}, max: {ret.max()}, mean: {ret.mean()}")
        if torch.isnan(adv).any():
            logger.error(f"[PPOAgent.optim_preprocess] adv contains NaN! NaN count: {torch.isnan(adv).sum()}, stats - min: {adv.min()}, max: {adv.max()}, mean: {adv.mean()}")
        if torch.isnan(old_log_prob).any():
            logger.error(f"[PPOAgent.optim_preprocess] old_log_prob contains NaN! NaN count: {torch.isnan(old_log_prob).sum()}, stats - min: {old_log_prob.min()}, max: {old_log_prob.max()}, mean: {old_log_prob.mean()}")
        if torch.isnan(old_val).any():
            logger.error(f"[PPOAgent.optim_preprocess] old_val contains NaN! NaN count: {torch.isnan(old_val).sum()}, stats - min: {old_val.min()}, max: {old_val.max()}, mean: {old_val.mean()}")

        act_dist, val = self.get_act_val(ob)
        
        # Log NaN tracking for value prediction
        if torch.isnan(val).any():
            logger.error(f"[PPOAgent.optim_preprocess] val (from critic) contains NaN! NaN count: {torch.isnan(val).sum()}, stats - min: {val.min()}, max: {val.max()}, mean: {val.mean()}")
        
        log_prob = action_log_prob(action, act_dist)
        
        # Log NaN tracking for log_prob
        if torch.isnan(log_prob).any():
            logger.error(f"[PPOAgent.optim_preprocess] log_prob contains NaN! NaN count: {torch.isnan(log_prob).sum()}, stats - min: {log_prob.min()}, max: {log_prob.max()}, mean: {log_prob.mean()}")
            logger.error(f"[PPOAgent.optim_preprocess] action stats - min: {action.min()}, max: {action.max()}, mean: {action.mean()}")
        
        entropy = action_entropy(act_dist, log_prob)
        
        # Log NaN tracking for entropy
        if torch.isnan(entropy).any():
            logger.error(f"[PPOAgent.optim_preprocess] entropy contains NaN! NaN count: {torch.isnan(entropy).sum()}, stats - min: {entropy.min()}, max: {entropy.max()}, mean: {entropy.mean()}")
        if not all([x.ndim == 1 for x in [val, log_prob]]):
            raise ValueError('val, log_prob should be 1-dim!')
        processed_data = dict(
            val=val,
            old_val=old_val,
            ret=ret,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            adv=adv,
            entropy=entropy
        )
        return processed_data

    def cal_loss(self, val, old_val, ret, log_prob, old_log_prob,
                 adv, entropy, *args, **kwargs):
        # Log NaN tracking for inputs to cal_loss
        if torch.isnan(val).any():
            logger.error(f"[PPOAgent.cal_loss] Input val contains NaN! NaN count: {torch.isnan(val).sum()}")
        if torch.isnan(ret).any():
            logger.error(f"[PPOAgent.cal_loss] Input ret contains NaN! NaN count: {torch.isnan(ret).sum()}")
        if torch.isnan(log_prob).any():
            logger.error(f"[PPOAgent.cal_loss] Input log_prob contains NaN! NaN count: {torch.isnan(log_prob).sum()}")
        if torch.isnan(old_log_prob).any():
            logger.error(f"[PPOAgent.cal_loss] Input old_log_prob contains NaN! NaN count: {torch.isnan(old_log_prob).sum()}")
        if torch.isnan(adv).any():
            logger.error(f"[PPOAgent.cal_loss] Input adv contains NaN! NaN count: {torch.isnan(adv).sum()}")
        if torch.isnan(entropy).any():
            logger.error(f"[PPOAgent.cal_loss] Input entropy contains NaN! NaN count: {torch.isnan(entropy).sum()}")
        
        entropy = torch.mean(entropy)
        
        # Log NaN tracking for mean entropy
        if torch.isnan(entropy):
            logger.error(f"[PPOAgent.cal_loss] Mean entropy is NaN!")
        
        vf_loss = self.cal_val_loss(val=val, old_val=old_val, ret=ret)
        
        # Log NaN tracking for vf_loss
        if torch.isnan(vf_loss):
            logger.error(f"[PPOAgent.cal_loss] vf_loss is NaN! val stats - min: {val.min()}, max: {val.max()}, mean: {val.mean()}")
            logger.error(f"[PPOAgent.cal_loss] ret stats - min: {ret.min()}, max: {ret.max()}, mean: {ret.mean()}")
        
        log_prob_diff = log_prob - old_log_prob
        
        # Log NaN tracking for log_prob difference
        if torch.isnan(log_prob_diff).any():
            logger.error(f"[PPOAgent.cal_loss] log_prob - old_log_prob contains NaN! NaN count: {torch.isnan(log_prob_diff).sum()}")
            logger.error(f"[PPOAgent.cal_loss] log_prob_diff stats - min: {log_prob_diff.min()}, max: {log_prob_diff.max()}, mean: {log_prob_diff.mean()}")
        
        ratio = torch.exp(log_prob_diff)
        
        # Log NaN tracking for ratio
        if torch.isnan(ratio).any():
            logger.error(f"[PPOAgent.cal_loss] ratio (exp of log_prob_diff) contains NaN! NaN count: {torch.isnan(ratio).sum()}")
            logger.error(f"[PPOAgent.cal_loss] ratio stats - min: {ratio.min()}, max: {ratio.max()}, mean: {ratio.mean()}")
            logger.error(f"[PPOAgent.cal_loss] log_prob_diff that caused NaN ratio - min: {log_prob_diff.min()}, max: {log_prob_diff.max()}, mean: {log_prob_diff.mean()}")
            logger.error(f"[PPOAgent.cal_loss] Extreme log_prob_diff values: min={log_prob_diff.min()}, max={log_prob_diff.max()}")
        
        surr1 = adv * ratio
        
        # Log NaN tracking for surr1
        if torch.isnan(surr1).any():
            logger.error(f"[PPOAgent.cal_loss] surr1 (adv * ratio) contains NaN! NaN count: {torch.isnan(surr1).sum()}")
        
        surr2 = adv * torch.clamp(ratio,
                                  1 - self.cfg.alg.clip_range,
                                  1 + self.cfg.alg.clip_range)
        
        # Log NaN tracking for surr2
        if torch.isnan(surr2).any():
            logger.error(f"[PPOAgent.cal_loss] surr2 contains NaN! NaN count: {torch.isnan(surr2).sum()}")
        
        pg_loss = -torch.mean(torch.min(surr1, surr2))
        
        # Log NaN tracking for pg_loss
        if torch.isnan(pg_loss):
            logger.error(f"[PPOAgent.cal_loss] pg_loss is NaN! surr1 stats - min: {surr1.min()}, max: {surr1.max()}, mean: {surr1.mean()}")
            logger.error(f"[PPOAgent.cal_loss] surr2 stats - min: {surr2.min()}, max: {surr2.max()}, mean: {surr2.mean()}")

        loss = pg_loss + vf_loss * self.cfg.alg.vf_coef

        loss = loss - entropy * self.cfg.alg.ent_coef
        
        # Log NaN tracking for final loss
        if torch.isnan(loss):
            logger.error(f"[PPOAgent.cal_loss] Final loss is NaN! pg_loss: {pg_loss}, vf_loss: {vf_loss}, entropy: {entropy}")
            logger.error(f"[PPOAgent.cal_loss] vf_coef: {self.cfg.alg.vf_coef}, ent_coef: {self.cfg.alg.ent_coef}")

        with torch.no_grad():
            approx_kl = 0.5 * torch.mean(torch.pow(old_log_prob - log_prob, 2))
            clr = torch.abs(ratio - 1.0) > self.cfg.alg.clip_range
            clip_frac = torch.mean(clr.float())
        return loss, pg_loss, vf_loss, ratio, entropy, approx_kl, clip_frac

    def cal_val_loss(self, val, old_val, ret):
        vf_loss = self.val_loss_criterion(val, ret)
        return vf_loss

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def save_model(self, wandb_run, is_best=False, step=None, eval=False):
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }
        save_model(data_to_save,
                   wandb_run,
                   is_best=is_best,
                   step=step,
                   eval=eval)

    def load_model(self, pretrain_model=None, eval=False):
        ckpt_data = load_ckpt_data(self.cfg.resume_id,
                                   project_name=self.cfg.logging.wandb.project,
                                   pretrain_model=pretrain_model, eval=eval)
        load_state_dict(self.actor,
                        ckpt_data.get('actor_state_dict', dict()))
        load_state_dict(self.critic,
                        ckpt_data.get('critic_state_dict', dict()))
        if pretrain_model is not None:
            return ckpt_data['step']
        if self.cfg.resume_optim and not self.cfg.test:
            self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        logger.info(f"Checkpoint step:{ckpt_data['step']}")
        return ckpt_data['step']

    def print_param_grad_status(self):
        logger.info('Requires Grad?')
        logger.info('================== Actor ================== ')
        for name, param in self.actor.named_parameters():
            print(f'{name}: {param.requires_grad}')
        logger.info('================== Critic ================== ')
        for name, param in self.critic.named_parameters():
            print(f'{name}: {param.requires_grad}')
