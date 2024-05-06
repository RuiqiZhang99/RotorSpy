import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import algos.rl.ppo_core as core
from algos.rl.utils import EpochLogger,setup_logger_kwargs, setup_pytorch_for_mpi, sync_params
from algos.rl.utils import mpi_avg_grads, mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from envs.LargeQuad_fake import DroneDock_Env
from utils.vehicle import Vehicle
import argparse
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def ppo_test(env_fn, 
        seed=0, 
        test_steps=5000, 
        logger_kwargs=dict(),
        load_model=False, # Don't load model when testing the pid controller
        ):
    # Set up logger and save configuration
    # logger = EpochLogger(**logger_kwargs)

    # Instantiate environment
    obs_dim = env_fn.observation_space.shape
    act_dim = env_fn.action_space.shape

    # Create actor-critic module
    if load_model:
        actor_critic_1 = torch.load('./algos/data/ppo/ppo_s{}/pyt_save/model.pt'.format(seed))

    # Prepare for interaction with environment
    start_time = time.time()
    obs_1, info = env_fn.reset()
    # obs_1, info = env.reset()
    ep_ret, ep_len = 0, 0
    
    time_line = np.zeros((test_steps, 1))
    cmd_history_1, cmd_history_2 = np.zeros((test_steps, 4)), np.zeros((test_steps, 4))
    pos_history_1, pos_history_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    ypr_history_1, ypr_history_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    angvel_history_1, angvel_history_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    vel_history_1, vel_history_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    desThrust_1, desThrust_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    desAngAcc_1, desAngAcc_2 = np.zeros((test_steps, 3)), np.zeros((test_steps, 3))
    fakeDwForce = np.zeros((test_steps, 3))
    
    
    for t in range(test_steps):
        if load_model:
            action_1 = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))[0]
        else:
            action_1 = np.zeros(4)
        
        time_line[t] = t * env_fn.dt
        cmd_history_1[t,:] = env_fn.cur_pwm_cmd_1
        pos_history_1[t,:] = env_fn.quadrotor_1._pos.to_array().squeeze()
        vel_history_1[t,:] = env_fn.quadrotor_1._vel.to_array().squeeze()
        ypr_history_1[t,:] = env_fn.quadrotor_1._ypr.squeeze()
        angvel_history_1[t,:] = env_fn.quadrotor_1._omega.to_array().squeeze()
        desThrust_1[t,:] = env_fn.thrustNormDes_1.to_array().squeeze()
        desAngAcc_1[t,:] = env_fn.angAccDes_1.to_array().squeeze()
        fakeDwForce[t,:] = env_fn.fake_downwashForce.to_array().squeeze()
        
        next_obs_1, reward_1, done, info = env_fn.step(action_1)
        # next_obs_1, reward, done, info = env.step(action_1)
        ep_ret += reward_1
        ep_len += 1
        
        # Update obs (critical!)
        obs_1 = next_obs_1
        test_ended = t==test_steps - 1

        if test_ended:
            if load_model:
                _ = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))[0]
            obs_1, info = env_fn.reset()
            # obs_1, info = env.reset()
            ep_ret, ep_len = 0, 0
    
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(8, 1, sharex=True)
    ax[0].plot(time_line, pos_history_1[:,0], label='pos_x')
    ax[0].plot(time_line, pos_history_1[:,1], label='pos_y')
    ax[0].plot(time_line, pos_history_1[:,2], label='pos_z')
    ax[0].plot(time_line, np.ones_like(pos_history_1)*env_fn.des_pos1[2], label='des_z')
    ax[1].plot(time_line, ypr_history_1[:,0], label='y')
    ax[1].plot(time_line, ypr_history_1[:,1], label='p')
    ax[1].plot(time_line, ypr_history_1[:,2], label='r')
    
    ax[2].plot(time_line, vel_history_1[:,0], label='vel_x')
    ax[2].plot(time_line, vel_history_1[:,1], label='vel_y')
    ax[2].plot(time_line, vel_history_1[:,2], label='vel_z')
    
    ax[3].plot(time_line, angvel_history_1[:,0], label='angvel_x')
    ax[3].plot(time_line, angvel_history_1[:,1], label='angvel_y')
    ax[3].plot(time_line, angvel_history_1[:,2], label='angvel_z')
    
    ax[4].plot(time_line, cmd_history_1[:,0], label='cmd_1')
    ax[4].plot(time_line, cmd_history_1[:,1], label='cmd_2')
    ax[4].plot(time_line, cmd_history_1[:,2], label='cmd_3')
    ax[4].plot(time_line, cmd_history_1[:,3], label='cmd_4')
    
    ax[5].plot(time_line, desThrust_1[:,0], label='desThrust_x')
    ax[5].plot(time_line, desThrust_1[:,1], label='desThrust_y')
    ax[5].plot(time_line, desThrust_1[:,2], label='desThrust_z')
    
    ax[6].plot(time_line, desAngAcc_1[:,0], label='desAngAcc_x')
    ax[6].plot(time_line, desAngAcc_1[:,1], label='desAngAcc_y')
    ax[6].plot(time_line, desAngAcc_1[:,2], label='desAngAcc_z')
    ax[7].plot(time_line, fakeDwForce[:,2], label='downwash')
    
    
    ax[0].set_ylabel('Pos [m]')
    ax[1].set_ylabel('Attitude [rad]')
    ax[2].set_ylabel('Vel [m/s]')
    ax[3].set_ylabel('AngVel [rad/s]')
    ax[4].set_ylabel('CMD')
    ax[5].set_ylabel('Desire Thrust')
    ax[6].set_ylabel('Desire Angular Acc.')
    ax[7].set_ylabel('Downwash')
    ax[7].set_xlabel('Time [s]')
    
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    ax[5].legend()
    ax[6].legend()
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=DroneDock_Env())
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    # wandb.init(project="real_world_learning", name=f"ppo-residual-ll", entity="hiperlab")
    env = DroneDock_Env(pid_control=True, highlevel_on=True)
    ppo_test(env_fn=env,
        seed=args.seed, 
        test_steps=args.steps, 
        logger_kwargs=logger_kwargs)
    
    # wandb.finish()