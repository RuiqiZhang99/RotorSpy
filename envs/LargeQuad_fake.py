from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from utils.vehicle import Vehicle
from utils.positioncontroller import PositionController
from utils.attitudecontroller import QuadAttiController, QuadAttiControllerNested
from utils.mixer import QuadcopterMixer
from utils.animate import animate_quadcopter_history, animate_multidrones_history
from utils.downwash import downwash, downwash_karan
from adap_drone_lowlevelctrl.utils import QuadState, Model
from adap_drone_lowlevelctrl.controller import AdapLowLevelControl
from pyplot3d.utils import ypr_to_R
import gym
from gym import spaces
import pandas as pd
from scipy.spatial.transform import Rotation as R
import random
import math

np.random.seed(0)

dt = 0.002  # sdifferent
endTime = 5.0

motInertia   = 0.0  #inertia of all rotating parts (motor + prop) [kg.m**2]
motTimeConst = 0.0  # time constant with which motor's speed responds [s]
motMinSpeed  = 0  #[rad/s]
motMaxSpeed_1 = 1385  #[rad/s]
motMaxSpeed_2 = 5000 #[rad/s]
stdDevTorqueDisturbance = 1e-3  # [N.m]

angVel_timeConst_xy_1 = 0.032 # For Large Quad
att_timeConst_xy_1 = angVel_timeConst_xy_1 * 2
angVel_timeConst_z_1 = angVel_timeConst_xy_1 * 5
att_timeConst_z_1 = angVel_timeConst_z_1 * 2

angVel_timeConst_xy_2 = 0.03 # For Large Quad
att_timeConst_xy_2 = angVel_timeConst_xy_2 * 2
angVel_timeConst_z_2 = angVel_timeConst_xy_2 * 5
att_timeConst_z_2 = angVel_timeConst_z_2 * 2

#time constants for the angle components:
timeConstAngleRP = 0.15  # [s]
timeConstAngleY  = 0.5  # [s]

timeConstRatesRP = 0.05  # [s]
timeConstRatesY  = 0.25   # [s]

posCtrlNatFreq_1 = 3.8  # rad/s #3.8
posCtrlDampingRatio_1 = 0.7  # -
posCtrlNatFreq_2 = 5  # rad/s #3.0
posCtrlDampingRatio_2 = 0.7  # -

mass_LQ, mass_MQ = 0.75, 0.33  # kg
Ixx_LQ, Ixx_MQ = 5.507e-3, 2.364e-4
Iyy_LQ, Iyy_MQ = 5.507e-3, 2.364e-4
Izz_LQ, Izz_MQ = 9.877e-3, 3.032e-4

omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0]))  # 1.4e-4 N.m/(rad/s)**2

armLength_1 = 0.166  # m
armLength_2 = 0.05785

inertiaMatrix_LQ = np.matrix([[Ixx_LQ, 0, 0], [0, Iyy_LQ, 0], [0, 0, Izz_LQ]])
inertiaMatrix_MQ = np.matrix([[Ixx_MQ, 0, 0], [0, Iyy_MQ, 0], [0, 0, Izz_MQ]])

stdDevTorqueDisturbance = 1e-3  # [N.m]
motSpeedSqrToThrust_LQ = 7.64e-6  # propeller coefficient
motSpeedSqrToTorque_LQ = 8.172e-8  # propeller coefficient

motSpeedSqrToThrust_MQ = 1.145e-7  # propeller coefficient
motSpeedSqrToTorque_MQ = 5.8613-10  # propeller coefficient

quadrotor_1 = Vehicle(mass_LQ, inertiaMatrix_LQ, armLength_1, omegaSqrToDragTorque, stdDevTorqueDisturbance)
quadrotor_2 = Vehicle(mass_MQ, inertiaMatrix_MQ, armLength_2, omegaSqrToDragTorque, stdDevTorqueDisturbance)

quadrotor_1.fastadd_quadmotor(motMinSpeed, motMaxSpeed_1, motSpeedSqrToThrust_LQ, motSpeedSqrToTorque_LQ, motTimeConst, motInertia)
# quadrotor_2.fastadd_quadmotor(motMinSpeed, motMaxSpeed_1, motSpeedSqrToThrust_LQ, motSpeedSqrToTorque_LQ, motTimeConst, motInertia)
quadrotor_2.fastadd_quadmotor(motMinSpeed, motMaxSpeed_2, motSpeedSqrToThrust_MQ, motSpeedSqrToTorque_MQ, motTimeConst, motInertia)

class DroneDock_Env(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}
    def __init__(self,
                 quadrotor_1 = quadrotor_1, # Large_quad
                 init_pos1 = np.array([0, 0, 0]),
                 des_pos1 = np.array([0, 0, 1.35]),
                 dt = dt,
                 miniquad_random_init = False,
                 highlevel_on = False,
                 norm_obs = True,
                 pid_control = False,
                 residual_rl = False,
                 output_folder='results',
                 real_time = 10.0,
                 ):

        #========================= Constants ==========================
        self.g = 9.8
        self.dt = dt
        self.miniquad_random_init = miniquad_random_init
        self.highlevel_on = highlevel_on
        self.max_pwm = 5.0
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
        self.output_folder = output_folder
        self.init_pos1 = init_pos1
        self.des_pos1 = des_pos1
        self.cur_pwm_cmd_1 = np.zeros(4)
        self.last_pwm_cmd_1 = np.zeros(4)
    
        self.thrustNormDes_1 = Vec3(0, 0, 0)
        self.angAccDes_1 = Vec3(0, 0, 0)     
        self.posControl_1 = PositionController(posCtrlNatFreq_1, posCtrlDampingRatio_1)
        # self.attController_1 = QuadAttiControllerNested(att_timeConst_xy_1, att_timeConst_z_1, angVel_timeConst_xy_1, angVel_timeConst_z_1)
        self.fake_downwashForce = Vec3(0, 0, 0)
        self.fake_quad_pos_2 = Vec3(1, 1, 0)
        self.attController_1 = QuadAttiController(att_timeConst_xy_1, att_timeConst_z_1)
        # self.attController_2 = QuadAttiController(att_timeConst_xy_2, att_timeConst_z_2)
        #=========== Create action and observation spaces ===============
        self.action_space = self.actionSpace()
        self.observation_space = self.observationSpace()
        
        #================= Reset the environment ========================
        self.getMixer()
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
        self.time_step = 0
        self.timestep_counter = 0
        self.fake_quad_pos_2 = Vec3(1, 1, 0)
        self.fake_downwashForce = Vec3(0, 0, 0)
        self.quadrotor_1.set_position(Vec3(self.init_pos1))
        self.quadrotor_1.set_velocity(Vec3(0, 0, 0))
        self.quadrotor_1.set_attitude(Rotation.identity())
        self.quadrotor_1._omega = Vec3(0,0,0)
        
        if self.miniquad_random_init:
            random_x, random_y = random.uniform(-1, 1), random.uniform(-1, 1)
            self.init_pos2 = np.array([random_x, random_y, 0])
        
        self.cur_state_1 = QuadState()
        
        initial_obs_1 = self.computeObs()
        # initial_obs_1 = self.computeObs()
        initial_info_1 = self.computeInfo()
        
        return initial_obs_1, initial_info_1
    
    #===================================================================================
    
    def NormAct2PWM(self,
                    norm_action
                        ):
        if self.residual_rl:
            spd_cmd = self.max_pwm * norm_action.squeeze()
        else:
            spd_cmd = 0.5*self.max_pwm + 0.5*self.max_pwm * norm_action.squeeze()
        
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
             ):
        
        self.time_step += 1
        
        # ------------------------- Downwash & Distrubance Calculate --------------------------------
        if self.time_step > 1000 and self.time_step < 2000:
            self.fake_quad_pos_2 = Vec3(1-(self.time_step-1000)*0.001, 1-(self.time_step-1000)*0.001, 1.5)
        elif self.time_step >= 2000:
            self.fake_quad_pos_2 = Vec3(0, 0, 1.5)
        
        self.fake_downwashForce = downwash_karan(self.fake_quad_pos_2 - self.quadrotor_1._pos)
        forceDistrub = Vec3(random.gauss(0, 0.01), random.gauss(0, 0.01), random.gauss(0, 0.2))
        torqueDistrub = Vec3(random.gauss(0, 0.03), random.gauss(0, 0.03), random.gauss(0, 0.01))
        
        # ----------------------------- High-Level Controller ---------------------------------------
        accDes_1 = self.posControl_1.get_acceleration_command(Vec3(self.des_pos1), self.quadrotor_1._pos, self.quadrotor_1._vel) 
        self.thrustNormDes_1 = accDes_1 + Vec3(0, 0, 9.81) - 0.8 * (self.fake_downwashForce/self.quadrotor_1._mass)
        self.angAccDes_1 = self.attController_1.get_angular_acceleration(self.quadrotor_1._att, self.quadrotor_1._omega)
        # self.angAccDes_1 = self.attController_1.get_angular_acceleration(self.thrustNormDes_1, self.quadrotor_1._att, self.quadrotor_1._omega)
        
        # -------------------------- End of High-level Controller ---------------------------------------
        
        if self.pid_controller:
            pwm_cmd_1 = self.mixer_1.get_motor_force_cmd(self.thrustNormDes_1, self.angAccDes_1)
        elif self.residual_rl:
            pwm_cmd_1 = self.NormAct2PWM(action_1) + self.mixer_1.get_motor_force_cmd(self.thrustNormDes_1, self.angAccDes_1)
        else:
            pwm_cmd_1 = self.NormAct2PWM(action_1)
        
        self.cur_pwm_cmd_1 = pwm_cmd_1
        
        self.quadrotor_1.run(self.dt, pwm_cmd_1, self.fake_downwashForce, forceDistrub, torqueDistrub)
        obs_1 = self.computeObs()
        reward_1 = self.computeReward()
        
        self.last_pwm_cmd_1 = pwm_cmd_1
        terminated = self.computeTerminated()
        truncated = self.computeTruncated()
        info_1 = self.computeInfo()
        
        self.timestep_counter += 1
        done = terminated or truncated
        return obs_1, reward_1, done, info_1
    
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
        
        obs_size = 20 if self.highlevel_on else 15    
        if self.norm_obs:
            obs_lower_bound = np.array(-1.0 * np.ones(obs_size))
            obs_upper_bound = np.array(+1.0 * np.ones(obs_size))
        else:
            obs_lower_bound = np.array(-self.max_obs * np.ones(obs_size))
            obs_upper_bound = np.array(+self.max_obs * np.ones(obs_size))
            
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    #===================================================================================
    def convert_vehState(self, veh_state, des_pos, fake_quad_pos_2):
        att_aray = np.array([veh_state.att[1], veh_state.att[2],
                             veh_state.att[3], veh_state.att[0]])
        rotation_matrix = R.from_quat(att_aray).as_matrix().reshape((9,), order="F")
        if self.highlevel_on:
            obs = np.concatenate((  veh_state.pos - fake_quad_pos_2.to_array().flatten(),
                                    rotation_matrix, # 9
                                    veh_state.omega, # 3
                                    np.array([veh_state.proper_acc[2]],dtype=np.float32),  # 1
                                    veh_state.cmd_bodyrates, # 3
                                    np.array([veh_state.cmd_collective_thrust],dtype=np.float32),  # 1
                                    ), axis=0).astype(np.float32)
        else:
            obs = np.concatenate((
                                    (des_pos-veh_state.pos),
                                    veh_state.ypr,
                                    veh_state.vel, # 3
                                    veh_state.omega,
                                    veh_state.proper_acc,
                                    ), axis=0).astype(np.float32)
        
        clipped_obs = np.clip(obs, -self.max_obs, self.max_obs)
        if self.norm_obs:
            normalized_obs = (1/self.max_obs)*clipped_obs
            return normalized_obs
        else:
            return clipped_obs
    
    
    def computeObs(self):
        
        self.cur_state_1.pos = self.quadrotor_1._pos.to_array().flatten()
        self.cur_state_1.att = self.quadrotor_1._att.to_array().flatten()
        self.cur_state_1.ypr = self.quadrotor_1._ypr.flatten()
        self.cur_state_1.omega = self.quadrotor_1._omega.to_array().flatten()
        self.cur_state_1.proper_acc = self.quadrotor_1._accel.to_array().flatten()
        self.cur_state_1.cmd_collective_thrust = self.thrustNormDes_1.z
        self.cur_state_1.cmd_bodyrates = self.angAccDes_1.to_array().flatten()
        
        obs_1 = self.convert_vehState(self.cur_state_1, self.des_pos1, self.fake_quad_pos_2)
        
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
        return obs_1


    #===================================================================================

    def computeReward(self, survival_reward=0.2, collision_penalty=-10):
        reward_1 = 0
        reward_1 += survival_reward \
            - 0.02 * np.linalg.norm(self.des_pos1 - self.cur_state_1.pos)\
            - 0.05 * np.linalg.norm(np.array([0, 0, 0]) - self.cur_state_1.ypr)\
            - 0.005 * np.linalg.norm((self.cur_pwm_cmd_1 - self.last_pwm_cmd_1), ord=1)
            
        if self.collision:
            reward_1 += collision_penalty
        return reward_1

    #===================================================================================

    def computeTerminated(self):
        return False
    
    #===================================================================================

    def computeTruncated(self):
        
        if self.cur_state_1.pos[0] > 2 or self.cur_state_1.pos[1] > 2 or self.cur_state_1.pos[2] > 2.5:
        # or self.cur_state_2.pos[0] > 2 or self.cur_state_2.pos[1] > 2 or self.cur_state_2.pos[2] > 2.5:
            return True
        if abs(self.cur_state_1.ypr[1]) > 0.3 or abs(self.cur_state_1.ypr[2]) > 0.3:
           # abs(self.cur_state_2.ypr[1]) > 0.3 or abs(self.cur_state_2.ypr[2]) > 0.3:
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
        info_1= None
        return info_1

    #===================================================================================
    
    def getMixer(self):
        self.mixer_1 = QuadcopterMixer(self.quadrotor_1._mass, self.quadrotor_1._inertia, self.quadrotor_1._armlength/(2**0.5), self.quadrotor_1.speedSqrToTorque/self.quadrotor_1.speedSqrToThrust)
        return
    