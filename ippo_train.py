import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import algos.rl.ppo_core as core
from algos.rl.utils import EpochLogger,setup_logger_kwargs, setup_pytorch_for_mpi, sync_params
from algos.rl.utils import mpi_avg_grads, mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from envs.multi_uav import DroneDock_Env
from utils.vehicle import Vehicle
import argparse

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


def ppo(env_fn, 
        actor_critic=core.MLPActorCritic, 
        ac_kwargs=dict(), 
        seed=0, 
        steps_per_epoch=20000, 
        epochs=500, 
        gamma=0.99, 
        clip_ratio=0.2, 
        pi_lr=1e-3,
        vf_lr=1e-3, 
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
    actor_critic_2 = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(actor_critic_1)
    sync_params(actor_critic_2)

    # Count variables
    var_counts_1 = tuple(core.count_vars(module) for module in [actor_critic_1.pi, actor_critic_1.v])
    var_counts_2 = tuple(core.count_vars(module) for module in [actor_critic_2.pi, actor_critic_2.v])
    logger.log('\nNumber of parameters of Main Quad: \t pi: %d, \t v: %d\n'%var_counts_1)
    logger.log('\nNumber of parameters of Mini Quad: \t pi: %d, \t v: %d\n'%var_counts_2)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buffer_1 = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    buffer_2 = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

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
    
    pi_optimizer_2 = Adam(actor_critic_2.pi.parameters(), lr=pi_lr)
    vf_optimizer_2 = Adam(actor_critic_2.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(actor_critic_1)
    logger.setup_pytorch_saver(actor_critic_2)

    def update(buffer, actor_critic, pi_optimizer, vf_optimizer, agent_id='1'):
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
        if agent_id == '1':
            logger.store(
                        LossPi_1=pi_l_old, 
                        LossV_1=v_l_old,
                        KL_1=kl, 
                        Entropy_1=ent, 
                        ClipFrac_1=cf,
                        DeltaLossPi_1=(loss_pi.item() - pi_l_old),
                        DeltaLossV_1=(loss_v.item() - v_l_old))
        elif agent_id == '2':
            logger.store(
                        LossPi_2=pi_l_old, 
                        LossV_2=v_l_old,
                        KL_2=kl, 
                        Entropy_2=ent, 
                        ClipFrac_2=cf,
                        DeltaLossPi_2=(loss_pi.item() - pi_l_old),
                        DeltaLossV_2=(loss_v.item() - v_l_old))
    

    # Prepare for interaction with environment
    start_time = time.time()
    obs_1, obs_2, info = env.reset()
    # obs_1, info = env.reset()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            action_1, value_1, logp_1 = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))
            action_2, value_2, logp_2 = actor_critic_2.step(torch.as_tensor(obs_2, dtype=torch.float32))

            next_obs_1, next_obs_2, reward, done, info = env.step(action_1, action_2)
            # next_obs_1, reward, done, info = env.step(action_1)
            ep_ret += reward
            ep_len += 1

            # save and log
            buffer_1.store(obs_1, action_1, reward, value_1, logp_1)
            buffer_2.store(obs_2, action_2, reward, value_2, logp_2)
            logger.store(VVals_1 = value_1)
            logger.store(VVals_2 = value_2)
            
            # Update obs (critical!)
            obs_1 = next_obs_1
            obs_2 = next_obs_2

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value_1, _ = actor_critic_1.step(torch.as_tensor(obs_1, dtype=torch.float32))
                    _, value_2, _ = actor_critic_2.step(torch.as_tensor(obs_2, dtype=torch.float32))
                else:
                    value_1, value_2 = 0, 0
                buffer_1.finish_path(value_1)
                buffer_2.finish_path(value_2)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs_1, obs_2, info = env.reset()
                # obs_1, info = env.reset()
                ep_ret, ep_len = 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update(buffer_1, actor_critic_1, pi_optimizer_1, vf_optimizer_1, agent_id='1')
        update(buffer_2, actor_critic_2, pi_optimizer_2, vf_optimizer_2, agent_id='2')

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        
        logger.log_tabular('VVals_1', with_min_and_max=True)
        logger.log_tabular('LossPi_1', average_only=True)
        logger.log_tabular('LossV_1', average_only=True)
        logger.log_tabular('DeltaLossPi_1', average_only=True)
        logger.log_tabular('DeltaLossV_1', average_only=True)
        logger.log_tabular('Entropy_1', average_only=True)
        logger.log_tabular('KL_1', average_only=True)
        logger.log_tabular('ClipFrac_1', average_only=True)
        
        logger.log_tabular('VVals_2', with_min_and_max=True)
        logger.log_tabular('LossPi_2', average_only=True)
        logger.log_tabular('LossV_2', average_only=True)
        logger.log_tabular('DeltaLossPi_2', average_only=True)
        logger.log_tabular('DeltaLossV_2', average_only=True)
        logger.log_tabular('Entropy_2', average_only=True)
        logger.log_tabular('KL_2', average_only=True)
        logger.log_tabular('ClipFrac_2', average_only=True)
        logger.dump_tabular()

if __name__ == '__main__':
    
    mass = 0.985  # kg
    Ixx = 4e-3
    Iyy = 8e-3
    Izz = 12e-3
    Ixy = 0
    Ixz = 0
    Iyz = 0
    omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0.00014]))  # N.m/(rad/s)**2
    armLength_1 = 0.4  # m
    armLength_2 = 0.2
    inertiaMatrix = np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    stdDevTorqueDisturbance = 1e-3  # [N.m]
    motSpeedSqrToThrust = 7.6e-6  # propeller coefficient
    motSpeedSqrToTorque = 1.1e-7  # propeller coefficient
    motInertia   = 15e-6  #inertia of all rotating parts (motor + prop) [kg.m**2]

    motTimeConst = 0.06  # time constant with which motor's speed responds [s]
    motMinSpeed  = 0  #[rad/s]
    motMaxSpeed  = 950  #[rad/s]
    TILT_ANGLE = np.deg2rad(15)
    
    quadrotor_1 = Vehicle(mass, inertiaMatrix, armLength_1, omegaSqrToDragTorque, stdDevTorqueDisturbance)
    quadrotor_2 = Vehicle(mass, inertiaMatrix, armLength_2, omegaSqrToDragTorque, stdDevTorqueDisturbance)
    quadrotor_1.fastadd_quadmotor(motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia, tilt_angle=TILT_ANGLE)
    quadrotor_2.fastadd_quadmotor(motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia, tilt_angle=TILT_ANGLE)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=DroneDock_Env(quadrotor_1, quadrotor_2))
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    env = DroneDock_Env(quadrotor_1, quadrotor_2, residual_rl=True)
    ppo(env_fn=env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hidden_dim]*args.layers), 
        gamma=args.gamma, 
        seed=args.seed, 
        steps_per_epoch=args.steps, 
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)