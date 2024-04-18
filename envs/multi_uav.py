from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from utils.vehicle import Vehicle
from utils.positioncontroller import PositionController
from utils.attitudecontroller import QuadcopterAttitudeControllerNested
from utils.mixer import QuadcopterMixer
from utils.animate import animate_quadcopter_history, animate_multidrones_history
from utils.downwash import downwash
from adap_drone_lowlevelctrl.utils import QuadState, Model
from adap_drone_lowlevelctrl.controller import AdapLowLevelControl
from pyplot3d.utils import ypr_to_R
import gym
from gym import spaces
import pandas as pd
from scipy.spatial.transform import Rotation as R
import random

np.random.seed(0)

#==============================================================================
# Define the simulation
#==============================================================================
dt = 0.002  # sdifferent
endTime = 10.0

#==============================================================================
# Define the Motors
#==============================================================================
motSpeedSqrToThrust = 7.6e-6  # propeller coefficient
motSpeedSqrToTorque = 1.1e-7  # propeller coefficient
motInertia   = 15e-6  #inertia of all rotating parts (motor + prop) [kg.m**2]

motTimeConst = 0.06  # time constant with which motor's speed responds [s]
motMinSpeed  = 0  #[rad/s]
motMaxSpeed  = 950  #[rad/s]
TILT_ANGLE = np.deg2rad(15)

#==============================================================================
# Define the disturbance
#==============================================================================
stdDevTorqueDisturbance = 1e-3  # [N.m]

#==============================================================================
# Define the attitude controller
#==============================================================================
#time constants for the angle components:
timeConstAngleRP = 0.15  # [s]
timeConstAngleY  = 0.5  # [s]

#gain from angular velocities
timeConstRatesRP = 0.05  # [s]
timeConstRatesY  = 0.25   # [s]

#==============================================================================
# Define the position controller
#==============================================================================
posCtrlNatFreq = 2  # rad/s
posCtrlDampingRatio = 0.7  # -


class DroneDock_Env(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}
    def __init__(self,
                 quadrotor_1, # Large_quad
                 quadrotor_2, # Mini_quad
                 init_pos1 = np.array([0, 0, 0]),
                 init_pos2 = np.array([1, 1, 0]),
                 des_pos1 = np.array([0, 0, 1.5]),
                 des_pos2 = np.array([0, 0, 1.6]),
                 dt = dt,
                 miniquad_random_init = True,
                 highlevel_on = True,
                 norm_obs = True,
                 pid_control = False,
                 residual_rl = False,
                 output_folder='results',
                 real_time = 10.0,
                 action_size = 4,
                 ):

        #========================= Constants ==========================
        self.g = 9.8
        self.rad_2_deg = 180 / np.pi
        self.deg_2_rad = np.pi / 180
        self.dt = dt
        self.ctrl_freq = int(1./self.dt)
        self.miniquad_random_init = miniquad_random_init
        self.highlevel_on = highlevel_on
        self.maxMotorSpd = 5000
        self.max_obs = 10.0
        self.max_timestep = int(real_time/self.dt)
        
        #=================== Multi-Drone Parameters ===================
        self.num_drones = 2
        self.collision = False
        self.norm_obs = norm_obs
        self.pid_controller = pid_control
        self.residual_rl = residual_rl
        
        #======================== Properties ==========================
        self.quadrotor_1 = quadrotor_1
        self.quadrotor_2 = quadrotor_2
        self.output_folder = output_folder
        self.init_pos1 = init_pos1
        self.init_pos2 = init_pos2
        self.des_pos1 = des_pos1
        self.des_pos2 = des_pos2
        
        self.posControl = PositionController(posCtrlNatFreq, posCtrlDampingRatio)
        self.attController = QuadcopterAttitudeControllerNested(timeConstAngleRP, timeConstAngleY, timeConstRatesRP, timeConstRatesY)
        # self.mixer_1 = QuadcopterMixer(self.quadrotor_1._mass, self.quadrotor_1._inertia, self.quadrotor_1._armlength/(2**0.5), self.quadrotor_1.speedSqrToTorque/self.quadrotor_1.speedSqrToThrust)
        # self.mixer_2 = QuadcopterMixer(self.quadrotor_2.mass, self.quadrotor_2._inertia, self.quadrotor_2._armlength/(2**0.5), self.quadrotor_2.speedSqrToTorque/self.quadrotor_2.speedSqrToThrust)
        
        #=========== Create action and observation spaces ===============
        self.action_space = self.actionSpace()
        self.observation_space = self.observationSpace()
        
        #================= Reset the environment ========================
        self.reset()
        '''
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()
        '''
    
    #===================================================================================

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        self.timestep_counter = 0

        self.quadrotor_1.set_position(Vec3(self.init_pos1))
        self.quadrotor_1.set_velocity(Vec3(0, 0, 0))
        self.quadrotor_1.set_attitude(Rotation.identity())
        self.quadrotor_1._omega = Vec3(0,0,0)
        
        if self.miniquad_random_init:
            random_x, random_y = random.uniform(-1, 1), random.uniform(-1, 1)
            self.init_pos2 = np.array([random_x, random_y, 0])
        self.quadrotor_2.set_position(Vec3(self.init_pos2))
        self.quadrotor_2.set_velocity(Vec3(0, 0, 0))
        self.quadrotor_2.set_attitude(Rotation.identity())
        self.quadrotor_2._omega = Vec3(0,0,0)
        
        self.cur_state_1, self.cur_state_2 = QuadState(), QuadState()
        
        initial_obs_1, initial_obs_2 = self.computeObs()
        # initial_obs_1 = self.computeObs()
        initial_info_1, initial_info_2 = self.computeInfo()
        
        return initial_obs_1, initial_obs_2, [initial_info_1, initial_info_2] # initial_obs_2, initial_info_1, initial_info_2
    
    #===================================================================================
    
    def NormAct2PWM(self,
                    norm_action
                        ):
        spd_cmd = 0.1 * self.maxMotorSpd * norm_action.squeeze()
        
        # hardcode to fit simulate drone model
        '''
        temp = spd_cmd[2]
        spd_cmd[2] = spd_cmd[1]
        spd_cmd[1] = temp
        '''
        
        return spd_cmd
    
    #===================================================================================

    def step(self,
             action_1,
             action_2,
             ):
        self.getMixer()
        if self.pid_controller:
            pwm_cmd_1 = self.mixer_1.get_motor_force_cmd(self.thrustNormDes_1, self.angVelDes_1)
            pwm_cmd_2 = self.mixer_2.get_motor_force_cmd(self.thrustNormDes_2, self.angVelDes_2)
        elif self.residual_rl:
            pwm_cmd_1 = self.NormAct2PWM(action_1) + self.mixer_1.get_motor_force_cmd(self.thrustNormDes_1, self.angVelDes_1)
            pwm_cmd_2 = self.NormAct2PWM(action_2) + self.mixer_2.get_motor_force_cmd(self.thrustNormDes_2, self.angVelDes_2)
        else:
            pwm_cmd_1 = self.NormAct2PWM(action_1)
            pwm_cmd_2 = self.NormAct2PWM(action_2)
        self.quadrotor_1.run(self.dt, pwm_cmd_1, spdCmd=True)
        self.quadrotor_2.run(self.dt, pwm_cmd_2, spdCmd=True)
        
        # Prepare the return values
        obs_1, obs_2 = self.computeObs()
        # obs_1 = self.computeObs()
        reward = self.computeReward()
        terminated = self.computeTerminated()
        truncated = self.computeTruncated()
        info_1, info_2 = self.computeInfo()
        info = [info_1, info_2]
        
        self.timestep_counter += 1
        done = terminated or truncated
        return obs_1, obs_2, reward, done, info
    
    #===================================================================================
    def render(self,
               mode='human',
               close=False
               ):
        return False

    #===================================================================================
    
    def actionSpace(self):
        act_size = 4
        act_lower_bound = np.array(-1 * np.ones(act_size))
        act_upper_bound = np.array(+1 * np.ones(act_size))
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    #===================================================================================

    def observationSpace(self):
        obs_size = 17
        if self.norm_obs and self.highlevel_on:
            obs_lower_bound = np.array(-1 * np.ones(obs_size))
            obs_upper_bound = np.array(+1 * np.ones(obs_size))
        elif self.highlevel_on:
            obs_lower_bound = np.array(-10 * np.ones(obs_size))
            obs_upper_bound = np.array(+10 * np.ones(obs_size))
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    #===================================================================================
    def convert_vehState(self, veh_state):
        att_aray = np.array([veh_state.att[1], veh_state.att[2],
                             veh_state.att[3], veh_state.att[0]])
        rotation_matrix = R.from_quat(att_aray).as_matrix().reshape((9,), order="F")
        obs = np.concatenate((
                                rotation_matrix, # 9
                                veh_state.omega, # 3
                                np.array([veh_state.proper_acc[2]],dtype=np.float32),  # 1
                                veh_state.cmd_bodyrates, # 3
                                np.array([veh_state.cmd_collective_thrust],dtype=np.float32),  # 1
                                ), axis=0).astype(np.float32)
        obs = np.clip(obs, -self.max_obs, self.max_obs)
        if self.norm_obs:
            normalized_obs = (1/self.max_obs)*obs
            return normalized_obs
        else:
            return obs
    
    
    def computeObs(self):
        
        accDes_1 = self.posControl.get_acceleration_command(Vec3(self.des_pos1), self.quadrotor_1._pos, self.quadrotor_1._vel)
        accDes_2 = self.posControl.get_acceleration_command(Vec3(self.des_pos2), self.quadrotor_2._pos, self.quadrotor_2._vel)
        
        self.thrustNormDes_1 = accDes_1 + Vec3(0, 0, 9.81)
        self.thrustNormDes_2 = accDes_2 + Vec3(0, 0, 9.81)
        
        # desired ang velocity 
        self.angVelDes_1 = self.attController.get_angular_velocity(self.thrustNormDes_1, self.quadrotor_1._att, self.quadrotor_1._omega)
        self.angVelDes_2 = self.attController.get_angular_velocity(self.thrustNormDes_2, self.quadrotor_2._att, self.quadrotor_2._omega)
        
        
        self.cur_state_1.pos = self.quadrotor_1._pos.to_array().flatten()
        self.cur_state_1.att = self.quadrotor_1._att.to_array().flatten()
        self.cur_state_1.ypr = self.quadrotor_1._ypr.flatten()
        self.cur_state_1.omega = self.quadrotor_1._omega.to_array().flatten()
        self.cur_state_1.proper_acc = self.quadrotor_1._accel.to_array().flatten()
        self.cur_state_1.cmd_collective_thrust = self.thrustNormDes_1.z
        self.cur_state_1.cmd_bodyrates = self.angVelDes_1.to_array().flatten()
        obs_1 = self.convert_vehState(self.cur_state_1)
        
        self.cur_state_2.pos = self.quadrotor_2._pos.to_array().flatten()
        self.cur_state_2.att = self.quadrotor_2._att.to_array().flatten()
        self.cur_state_2.ypr = self.quadrotor_2._ypr.flatten()
        self.cur_state_2.omega = self.quadrotor_2._omega.to_array().flatten()
        self.cur_state_2.proper_acc = self.quadrotor_2._accel.to_array().flatten()
        self.cur_state_2.cmd_collective_thrust = self.thrustNormDes_2.z
        self.cur_state_2.cmd_bodyrates = self.angVelDes_2.to_array().flatten()
        
        
        obs_1, obs_2 = self.convert_vehState(self.cur_state_1), self.convert_vehState(self.cur_state_2)
        
        
        '''
        obs_1 = np.hstack((self.quadrotor_1._att.to_array().flatten(),
                           self.quadrotor_1._omega.to_array().flatten(),
                           self.quadrotor_1._accel.to_array().flatten(),
                           thrustNormDes_1.z,
                           angVelDes_1.to_array().flatten()
                           ))
        
        obs_2 = np.hstack((self.quadrotor_2._att.to_array().flatten(),
                           self.quadrotor_2._omega.to_array().flatten(),
                           self.quadrotor_2._accel.to_array().flatten(),
                           thrustNormDes_2.z,
                           angVelDes_2.to_array().flatten()
                           ))
        '''
        return obs_1, obs_2


    #===================================================================================

    def computeReward(self, survival_reward=1, collision_penalty=-10):
        reward = 0
        reward += survival_reward \
            - 0.1 * np.linalg.norm(self.des_pos1 - self.cur_state_1.pos)\
            - 0.05 * np.linalg.norm([1, 0, 0, 0] - self.cur_state_1.att)\
            - 0.01 * np.linalg.norm([0, 0, 0] - self.cur_state_1.vel)\
            # - 0.01 * np.linalg.norm([0, 0, 0] - self.cur_state_1.omega)\
        '''
            - 0.1 * np.linalg.norm(self.des_pos2 - self.cur_state_2.pos)\
            - 0.05 * np.linalg.norm([1, 0, 0, 0] - self.cur_state_2.att)\
            - 0.01 * np.linalg.norm([0, 0, 0] - self.cur_state_2.vel)\
            - 0.01 * np.linalg.norm([0, 0, 0] - self.cur_state_2.omega)
        '''
        if self.collision:
            reward += collision_penalty
        return reward

    #===================================================================================

    def computeTerminated(self):
        return False
    
    #===================================================================================

    def computeTruncated(self):
        
        if self.cur_state_1.pos[0] > 2 or self.cur_state_1.pos[1] > 2 or self.cur_state_1.pos[2] > 2.5 or \
            self.cur_state_2.pos[0] > 2 or self.cur_state_2.pos[1] > 2 or self.cur_state_2.pos[2] > 2.5:
            return True
        if abs(self.cur_state_1.ypr[1]) > 0.3 or abs(self.cur_state_1.ypr[2]) > 0.3 or\
           abs(self.cur_state_2.ypr[1]) > 0.3 or abs(self.cur_state_2.ypr[2]) > 0.3:
            return True
        if self.timestep_counter > self.max_timestep:
            return True
        else:
            return False

    #===================================================================================

    def computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        info_1, info_2 = None, None
        return info_1, info_2

    #===================================================================================
    
    def getMixer(self):
        self.mixer_1 = QuadcopterMixer(self.quadrotor_1._mass, self.quadrotor_1._inertia, self.quadrotor_1._armlength/(2**0.5), self.quadrotor_1.speedSqrToTorque/self.quadrotor_1.speedSqrToThrust)
        self.mixer_2 = QuadcopterMixer(self.quadrotor_2._mass, self.quadrotor_2._inertia, self.quadrotor_2._armlength/(2**0.5), self.quadrotor_2.speedSqrToTorque/self.quadrotor_2.speedSqrToThrust)
        return
    