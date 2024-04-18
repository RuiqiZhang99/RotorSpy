import argparse
import math
#
import os

import numpy as np
import torch
# from flightgym import QuadrotorEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy  # replace MlpPolicy

from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy
import wandb
from rpg_baselines.torch.common.util import plot3d_traj, traj_rollout
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--log", type=int, default=1, help="Log the training or not")
    return parser

def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(open("./configs/rma_config.yaml", "r"))

    if not args.train:
        cfg["simulation"]["num_envs"] = 1

    # create training environment
    # TODO: Replace Env with MultipleDrone Env, delete Env Wrapper
    train_env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"

    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs
    
    latent_size = cfg['observation_space']['latent_size']
    

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    #
    rotor_ctrl = cfg["simulation"]["rotor_ctrl"]
    if args.train:

        total_timesteps=int(20 * 1e7)
        train_env.setTotalTimestep(total_timesteps)
        eval_env.setTotalTimestep(total_timesteps)
        

        n_steps = 500
        
        wandb.init(project="real_world_learning", name=f"run_{args.trial}", entity="hiperlab", config=cfg)
        
        model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
            activation_fn= torch.nn.Tanh, #torch.nn.ReLU,
            net_arch=[128, 128, latent_size, dict(pi=[256, 256], vf=[512, 512])],
            log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=n_steps,
            ent_coef=0.0,
            vf_coef=0.5,
            min_BC_coef=0.1,
            BC_alpha=0.999,
            max_grad_norm=0.5,
            batch_size=25000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            env_cfg=cfg,
            verbose=1,
            seed = args.seed,
        )

        try:
            model.learn(total_timesteps=total_timesteps,
                        log_interval=(10, int(1e6/(n_steps*old_num_envs))))

            wandb.finish()
        except KeyboardInterrupt:
            print("Keyboard interruption!")

            wandb.finish()
        wandb.finish()

    else:
        # os.system(os.environ["FLIGHTMARE_PATH"] +
        #                   "/flightrender/RPG_Flightmare.x86_64 &")
        #
        weight = rsg_root + "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        # Create policy object
        policy = MlpPolicy(**saved_variables["data"])
        #
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        # Load weights
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)
        # 
        eval_env.load_rms(env_rms)
        # test_policy(eval_env, policy, render=args.render)
        traj_df = traj_rollout(eval_env,policy)
        traj_df.to_csv("./phase1_test_traj_{0}_{1:05d}.csv".format(args.trial,args.iter))
        

if __name__ == "__main__":
    main()
