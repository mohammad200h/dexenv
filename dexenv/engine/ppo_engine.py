import time
import torch
import wandb
from dataclasses import dataclass
from itertools import count
from loguru import logger

from dexenv.engine.base_engine import BaseEngine
from dexenv.utils.common import stack_data
from dexenv.utils.dataset import DictDataset
from dexenv.utils.gae import cal_gae
from dexenv.utils.info_util import TIMEOUT_KEY
from dexenv.utils.info_util import aggregate_traj_info
from dexenv.utils.info_util import info_has_key
from dexenv.utils.torch_utils import swap_flatten_leading_axes


@dataclass
class PPOEngine(BaseEngine):

    def train(self):
        for iter_t in count():
            if iter_t % self.cfg.logging.eval_interval == 0 and self.cfg.alg.run_eval:
                self.do_eval()
            traj, rollout_time = self.rollout_once(sample=True,
                                                   get_last_val=True,
                                                   reset_first=self.cfg.alg.reset_first_in_rollout,
                                                   time_steps=self.cfg.alg.train_rollout_steps)
            optim_infos = self.train_once(traj)
            if iter_t % self.cfg.logging.log_interval == 0:
                t1 = time.perf_counter()
                train_log_info = self.get_train_log(optim_infos, traj)
                train_log_info['train/optim_time'] = t1 - self.optim_stime
                train_log_info['train/rollout_time'] = rollout_time
                wandb.log(train_log_info, step=self.cur_step)
            if self.cur_step > self.cfg.alg.max_steps:
                break
            if iter_t % self.cfg.logging.ckpt_interval == 0:
                self.agent.save_model(is_best=self._train_is_best,
                                      step=self.cur_step,
                                      wandb_run=self.wandb_runs,
                                      eval=False)

    def train_once(self, traj):
        self.optim_stime = time.perf_counter()
        self.cur_step += traj.total_steps
        rollout_dataloader = self.traj_preprocess(traj)
        optim_infos = []
        for oe in range(self.cfg.alg.opt_epochs):
            for batch_ndx, batch_data in enumerate(rollout_dataloader):
                optim_info = self.agent.optimize(batch_data)
                optim_infos.append(optim_info)
        return optim_infos

    def traj_preprocess(self, traj):
        action_infos = traj.action_infos
        vals = stack_data([ainfo['val'] for ainfo in action_infos])
        log_prob = stack_data([ainfo['log_prob'] for ainfo in action_infos])
        
        # Log NaN tracking for vals
        if torch.isnan(vals).any():
            logger.error(f"[PPOEngine.traj_preprocess] vals contains NaN! NaN count: {torch.isnan(vals).sum()}, stats - min: {vals.min()}, max: {vals.max()}, mean: {vals.mean()}")
        
        adv = self.cal_advantages(traj)
        
        # Log NaN tracking for adv after cal_advantages
        if torch.isnan(adv).any():
            logger.error(f"[PPOEngine.traj_preprocess] adv after cal_advantages contains NaN! NaN count: {torch.isnan(adv).sum()}, stats - min: {adv.min()}, max: {adv.max()}, mean: {adv.mean()}")
        
        ret = adv + vals
        
        # Log NaN tracking for ret after computation
        if torch.isnan(ret).any():
            logger.error(f"[PPOEngine.traj_preprocess] ret (adv + vals) contains NaN! NaN count: {torch.isnan(ret).sum()}, stats - min: {ret.min()}, max: {ret.max()}, mean: {ret.mean()}")
            logger.error(f"[PPOEngine.traj_preprocess] adv stats - min: {adv.min()}, max: {adv.max()}, mean: {adv.mean()}")
            logger.error(f"[PPOEngine.traj_preprocess] vals stats - min: {vals.min()}, max: {vals.max()}, mean: {vals.mean()}")
        
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / (adv_std + 1e-8)
        
        # Log NaN tracking for adv after normalization
        if torch.isnan(adv).any():
            logger.error(f"[PPOEngine.traj_preprocess] adv after normalization contains NaN! NaN count: {torch.isnan(adv).sum()}, stats - min: {adv.min()}, max: {adv.max()}, mean: {adv.mean()}")
            logger.error(f"[PPOEngine.traj_preprocess] adv_mean: {adv_mean}, adv_std: {adv_std}, adv_std + 1e-8: {adv_std + 1e-8}")
        data = dict(
            ob=swap_flatten_leading_axes(traj.obs),
            action=swap_flatten_leading_axes(traj.actions),
            ret=swap_flatten_leading_axes(ret),
            adv=swap_flatten_leading_axes(adv),
            log_prob=swap_flatten_leading_axes(log_prob),
            val=swap_flatten_leading_axes(vals)
        )

        rollout_dataset = DictDataset(**data)

        dataloader = self.get_dataloader(rollout_dataset,
                                         self._get_batch_size(rollout_dataset))
        return dataloader

    def cal_advantages(self, traj, start_time=None):
        rewards = traj.rewards
        dones = traj.dones
        action_infos = traj.action_infos
        vals = stack_data([ainfo['val'] for ainfo in action_infos])
        
        # Log NaN tracking for rewards and vals before GAE
        if torch.isnan(rewards).any():
            logger.error(f"[PPOEngine.cal_advantages] rewards contains NaN! NaN count: {torch.isnan(rewards).sum()}, stats - min: {rewards.min()}, max: {rewards.max()}, mean: {rewards.mean()}")
        if torch.isnan(vals).any():
            logger.error(f"[PPOEngine.cal_advantages] vals contains NaN! NaN count: {torch.isnan(vals).sum()}, stats - min: {vals.min()}, max: {vals.max()}, mean: {vals.mean()}")
        
        traj_infos = traj.infos
        if info_has_key(traj_infos, TIMEOUT_KEY):
            timeout_info = aggregate_traj_info(traj_infos, TIMEOUT_KEY)
        else:
            timeout_info = None
        if start_time is not None:
            rewards = rewards[start_time:]
            dones = dones[start_time:]
            vals = vals[start_time:]
        last_val = traj.extra_data['last_val']
        
        # Log NaN tracking for last_val
        if torch.isnan(last_val).any():
            logger.error(f"[PPOEngine.cal_advantages] last_val contains NaN! NaN count: {torch.isnan(last_val).sum()}, stats - min: {last_val.min()}, max: {last_val.max()}, mean: {last_val.mean()}")
        
        adv = cal_gae(gamma=self.cfg.alg.rew_discount,
                      lam=self.cfg.alg.gae_lambda,
                      rewards=rewards,
                      value_estimates=vals,
                      last_value=last_val,
                      dones=dones,
                      timeout=timeout_info)
        
        # Log NaN tracking for adv after cal_gae
        if torch.isnan(adv).any():
            logger.error(f"[PPOEngine.cal_advantages] adv after cal_gae contains NaN! NaN count: {torch.isnan(adv).sum()}, stats - min: {adv.min()}, max: {adv.max()}, mean: {adv.mean()}")
        
        return adv
