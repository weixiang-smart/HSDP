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
from quest.utils.logger import Logger
import json

import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OmegaConf.register_new_resolver("eval", eval, replace=True)


# @hydra.main(config_path="config", config_name='evaluate', version_base=None)
@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    
    # create model
    # save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True)
    # os.makedirs(save_dir)

    checkpoint_path = 'experiments/libero/LIBERO_90/diffusion_policy/final/block_32/0/stage_1/multitask_model_epoch_0020.pth'

    # if cfg.checkpoint_path is None:
    #     # Basically if you don't provide a checkpoint path it will automatically find one corresponding
    #     # to the experiment/variant name you provide
    #     checkpoint_path, _ = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
    #     checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    # else:
    #     checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
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


    dataset = instantiate(cfg.task.dataset)
    model.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    

    for param in model.diffusion_model.parameters():
            param.requires_grad = False
    
    if cfg.rollout.enabled:
        env_runner = instantiate(cfg.task.env_runner)
    
    logger = Logger(train_cfg.log_interval)

    feature_list = []
    label_list = []

    for epoch in range(0, 1):
    # for epoch in range(0, train_cfg.n_epochs + 1):
        t0 = time.time()
        training_loss = 0.0

        for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
            
            data = utils.map_tensor_to_device(data, device)
            action, middle_feature = model.sample_actions(data)

            label = np.expand_dims(data['task_id'].cpu().numpy(), axis=1)
            middle_feature = middle_feature.view(middle_feature.shape[0], -1)
            middle_feature_numpy = middle_feature.detach().cpu().numpy()

            feature_list.append(middle_feature_numpy)
            label_list.append(label)

        feature_array = np.concatenate(feature_list,axis=0)
        label_array = np.concatenate(label_list,axis=0)

        np.save('feature_array_20.npy', feature_array)
        np.save('label_array_20.npy', label_array)

        # print(feature_array.shape, label_array.shape)

        # mask = np.isin(label_array.squeeze(),[0,1,2])
        # print(mask.shape)

        # feature_array_used = feature_array[mask]
        # label_array_used = label_array[mask].squeeze()

    #     print(feature_array_used.shape, label_array_used.shape)

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(feature_array_used)

    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label_array_used, 
    #                     cmap='tab10', alpha=0.6, edgecolors='w', linewidth=0.5)
    # plt.colorbar(scatter, label='Class Label')
    # plt.title('T-SNE Visualization of Feature Embeddings')
    # plt.xlabel('TSNE Component 1')
    # plt.ylabel('TSNE Component 2')
    # plt.show()

            # flattened_numpy = middle_feature_numpy.view(middle_feature_numpy.shape[0], -1)

            # print(middle_feature_numpy.shape)

            #print(label)
            
            # for low_model_optimizer in low_model_optimizers:
            #     low_model_optimizer.zero_grad()

            # with torch.autograd.set_detect_anomaly(False):
            #     with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_cfg.use_amp):
            #         loss, info = low_model.compute_loss(data)
            
            #     scaler.scale(loss).backward()
            
            # for low_model_optimizer in low_model_optimizers:
            #     scaler.unscale_(low_model_optimizer)
            # if train_cfg.grad_clip is not None:
            #     grad_norm = nn.utils.clip_grad_norm_(
            #         low_model.parameters(), train_cfg.grad_clip
            #     )

            # for low_model_optimizer in low_model_optimizers:
            #     scaler.step(low_model_optimizer)
            
            # scaler.update()

            # info.update({
            #     'epoch': epoch
            # })
            # if train_cfg.grad_clip is not None:
            #     info.update({
            #         "grad_norm": grad_norm.item(),
            #     })  
            # info = {cfg.logging_folder: info}
            # training_loss += loss.item()
            # steps += 1
            # logger.update(info, steps)

            # if train_cfg.cut and idx > train_cfg.cut:
            #     break



    # env_runner = instantiate(cfg.task.env_runner)
    
    # print('Saving to:', save_dir)
    # print('Running evaluation...')

    # def save_video_fn(video_chw, env_name, idx):
    #     video_dir = os.path.join(save_dir, 'videos', env_name)
    #     os.makedirs(video_dir, exist_ok=True)
    #     save_path = os.path.join(video_dir, f'{idx}.mp4')
    #     clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
    #     clip.write_videofile(save_path, fps=24, verbose=False, logger=None)

    # if train_cfg.do_profile:
    #     profiler = Profiler()
    #     profiler.start()
    # rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=train_cfg.use_tqdm, save_video_fn=save_video_fn)
    # if train_cfg.do_profile:
    #     profiler.stop()
    #     profiler.print()
    # print(
    #     f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
    #         | environments solved: {rollout_results['rollout']['environments_solved']}")

    # with open(os.path.join(save_dir, 'data.json'), 'w') as f:
    #     json.dump(rollout_results, f)


if __name__ == "__main__":
    main()
