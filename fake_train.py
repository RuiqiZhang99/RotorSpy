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

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

def learning_curve_display(epoch, last_show_num, logger, eval_rew_list):
    eval_rew_list.append(np.mean(logger.epoch_dict['EpRet']))
    if epoch / last_show_num > 1.1:
        plt.cla()
        # plt.title(track + train_mode, loc='center')
        plt.plot(eval_rew_list, label="Rewards")
        plt.legend()
        plt.pause(0.01)
        last_show_num = epoch
    return eval_rew_list, last_show_num

def ppo(env_fn, 
        actor_critic=core.MLPActorCritic, 
        ac_kwargs=dict(), 
        seed=0, 
        steps_per_epoch=5000, 
        epochs=1000, 
        gamma=0.99, 
        clip_ratio=0.2, 
        pi_lr=3e-4,
        vf_lr=3e-4, 
        train_pi_iters=2, 
        train_v_iters=2, 
        lam=0.97, 
        max_ep_len=5000,
        target_kl=0.01, 
        logger_kwargs=dict(), 
        save_freq=10):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    obs_dim = env_fn.observation_space.shape
    act_dim = env_fn.action_space.shape

    # Create actor-critic module
    actor_critic_1 = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    logger.setup_pytorch_saver(actor_critic_1)

    # Sync params across processes
    sync_params(actor_critic_1)

    # Count variables
    var_counts_1 = tuple(core.count_vars(module) for module in [actor_critic_1.pi, actor_critic_1.v])
    logger.log('\nNumber of parameters of Main Quad: \t pi: %d, \t v: %d\n'%var_counts_1)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buffer_1 = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, actor_critic):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, actor_critic):
        obs, ret = data['obs'], data['ret']
        return ((actor_critic.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer_1 = Adam(actor_critic_1.pi.parameters(), lr=pi_lr)
    vf_optimizer_1 = Adam(actor_critic_1.v.parameters(), lr=vf_lr)
    
    # Set up model saving
    logger.setup_pytorch_saver(actor_critic_1)

    def update(buffer, actor_critic, pi_optimizer, vf_optimizer):
        data = buffer.get()
        pi_l_old, pi_info_old = compute_loss_pi(data, actor_critic)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, actor_critic).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, actor_critic)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(actor_critic.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, actor_critic)
            loss_v.backward()
            mpi_avg_grads(actor_critic.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(
                    LossPi_1=pi_l_old, 
                    LossV_1=v_l_old,
                    KL_1=kl, 
                    Entropy_1=ent, 
                    ClipFrac_1=cf,
                    DeltaLossPi_1=(loss_pi.item() - pi_l_old),
                    DeltaLossV_1=(loss_v.item() - v_l_old))
            
    last_show_num = 1
    eval_rew_list = []

    # Prepare for interaction with environment
    start_time = time.time()
    obs_1, info = env_fn.reset()
    # obs_1, info = env.reset()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            action_1, value_1, logp_1 = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))

            next_obs_1, reward_1, done, info = env_fn.step(action_1)
            # next_obs_1, reward, done, info = env.step(action_1)
            ep_ret += reward_1
            ep_len += 1

            # save and log
            buffer_1.store(obs_1, action_1, reward_1, value_1, logp_1)
            logger.store(VVals_1 = value_1)
            
            # Update obs (critical!)
            obs_1 = next_obs_1

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value_1, _ = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))
                else:
                    value_1 = 0
                buffer_1.finish_path(value_1)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs_1, info = env_fn.reset()
                # obs_1, info = env.reset()
                ep_ret, ep_len = 0, 0


        # Save the best model
        if epoch > 0:
            if np.mean(logger.epoch_dict['EpRet']) > max(eval_rew_list):
                print('Find the Best Performance Model !!!')
                logger.save_state({'env': env})
                print('Saved!')

        # Perform PPO update!
        update(buffer_1, actor_critic_1, pi_optimizer_1, vf_optimizer_1)
        eval_rew_list,  last_show_num = learning_curve_display(epoch, last_show_num, logger, eval_rew_list)

        
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('VVals_1', average_only=True)
        logger.log_tabular('LossPi_1', average_only=True)
        logger.log_tabular('LossV_1', average_only=True)
        # logger.log_tabular('DeltaLossPi_1', average_only=True)
        # logger.log_tabular('DeltaLossV_1', average_only=True)
        logger.log_tabular('Entropy_1', average_only=True)
        logger.log_tabular('KL_1', average_only=True)
        logger.log_tabular('ClipFrac_1', average_only=True)
        
        logger.dump_tabular()

if __name__ == '__main__':
    
    env = DroneDock_Env(pid_control=True, highlevel_on=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='pid')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    # wandb.init(project="real_world_learning", name=f"ppo-residual-ll", entity="hiperlab")
    
    ppo(env_fn=args.env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hidden_dim]*args.layers), 
        gamma=args.gamma, 
        seed=args.seed, 
        steps_per_epoch=args.steps, 
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)
    
    # wandb.finish()