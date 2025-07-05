import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import quest.utils.utils as utils
from pyinstrument import Profiler
from moviepy.editor import ImageSequenceClip
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="config", config_name='load_model', version_base=None)
def load_model(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    
    # # create model
    # save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True)
    # os.makedirs(save_dir)

    if cfg.checkpoint_path is None:
        # Basically if you don't provide a checkpoint path it will automatically find one corresponding
        # to the experiment/variant name you provide
        checkpoint_path, _ = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)

    # print(checkpoint_path)
    state_dict = utils.load_state(checkpoint_path)
    
    if 'config' in state_dict:
        print('autoloading based on saved parameters')
        model = instantiate(state_dict['config']['algo']['policy'], 
                            shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy,
                            shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict['model'])

    print(type(model))

    return model


if __name__ == "__main__":
    model = load_model()
    print(type(model))
