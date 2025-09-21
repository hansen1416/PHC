import glob
import os
import sys
import pdb
import os.path as osp
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.getcwd())

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from isaacgym import gymapi
from isaacgym import gymutil

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
# from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from phc.utils.flags import flags

import numpy as np
import copy
import torch
import wandb

from learning import im_amp
from learning import im_amp_players
from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder
from learning import amp_network_mcp_builder
from learning import amp_network_pnn_builder

from env.tasks import humanoid_amp_task
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

class AlgoObserver:
    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass



class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


@hydra.main(
    version_base=None,
    config_path="../phc/data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))

    # print(cfg)

    set_np_formatting()

        # cfg, cfg_train, logdir = load_cfg(args)
    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d
    
    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    cfg.train = False

    project_name = cfg.get("project_name", "egoquest")

    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    cfg.output_path = os.path.join("output", "HumanoidIm", "phc_kp_mcp_iccv")

    # # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs

    # print(cfg_train)
    # print(cfg.output_path)
    # print(cfg_train["params"]["config"]['name'])

    path = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + '.pth')
    if osp.exists(path):
        cfg_train["params"]["load_path"] = path
        cfg_train["params"]["load_checkpoint"] = True
    else:
        print(path)
        raise Exception("no file to resume!!!!")
    
    # print(cfg_train["params"]["load_path"])

    algo_observer = RLGPUAlgoObserver()

    print(algo_observer)


if __name__ == '__main__':
    main()
