from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from utils.vehicle import Vehicle
from utils.positioncontroller import PositionController
from utils.attitudecontroller import QuadAttiControllerNested
from utils.mixer import QuadcopterMixer
from utils.animate import animate_quadcopter_history, animate_multidrones_history
from utils.downwash import downwash
from adap_drone_lowlevelctrl.utils import QuadState, Model
from adap_drone_lowlevelctrl.controller import AdapLowLevelControl
from pyplot3d.utils import ypr_to_R

import pandas as pd

np.random.seed(0)

#==============================================================================
# Define the simulation
#==============================================================================
dt = 0.002  # sdifferent
endTime = 10

#==============================================================================
# Define the vehicle
#==============================================================================
mass = 0.985  # kg
Ixx = 4e-3
Iyy = 8e-3
Izz = 12e-3
Ixy = 0
Ixz = 0
Iyz = 0
omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0.00014]))  # N.m/(rad/s)**2
armLength_1 = 0.166  # m
armLength_2 = 0.058


#==============================================================================
# Define the Motors
#==============================================================================
motSpeedSqrToThrust = 7.6e-6  # propeller coefficient
motSpeedSqrToTorque = 1.1e-7  # propeller coefficient
motInertia   = 15e-6  #inertia of all rotating parts (motor + prop) [kg.m**2]

motTimeConst = 0.06  # time constant with which motor's speed responds [s]
motMinSpeed  = 0  #[rad/s]
motMaxSpeed  = 950  #[rad/s]

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
disablePositionControl = False
posCtrlNatFreq = 2  # rad/s
posCtrlDampingRatio = 0.7  # -

#==============================================================================
# Define RL lowlevel controller
#==============================================================================
low_level_controller = AdapLowLevelControl()
# Initialize our quadrotor's state
cur_state = QuadState()
# Set the maximum motor speed for this quadcopter model in RPM
# Note: because the controller can adapt down to motors-level, this highest-rpm 
# can be gained directly from motor datasheet without measurements from the experiments.
low_level_controller.set_max_motor_spd(motMaxSpeed)


#==============================================================================
# Compute all things:
#==============================================================================

inertiaMatrix = np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
quadrocopter_1 = Vehicle(mass, inertiaMatrix, armLength_1, omegaSqrToDragTorque, stdDevTorqueDisturbance)
quadrocopter_2 = Vehicle(mass, inertiaMatrix, armLength_2, omegaSqrToDragTorque, stdDevTorqueDisturbance)

# Our quadcopter model as    
#           
#           x
#           ^
#   (-)mot3 | mot0(+)
#           |
#     y<----+-----
#           |
#   (+)mot2 | mot1(-)
# motor_pos = armLength/(2**0.5)

quadrocopter_1.fastadd_quadmotor(motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
quadrocopter_2.fastadd_quadmotor(motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)

posControl = PositionController(posCtrlNatFreq, posCtrlDampingRatio)
attController = QuadAttiControllerNested(timeConstAngleRP, timeConstAngleY, timeConstRatesRP, timeConstRatesY)
mixer_1 = QuadcopterMixer(mass, inertiaMatrix, armLength_1/(2**0.5), motSpeedSqrToTorque/motSpeedSqrToThrust)
mixer_2 = QuadcopterMixer(mass, inertiaMatrix, armLength_2/(2**0.5), motSpeedSqrToTorque/motSpeedSqrToThrust)

#==============================================================================
# Set UAV No.1 and No.2
#==============================================================================
desPos_1 = Vec3(0, 0, 1)

quadrocopter_1.set_position(Vec3(0, 0, 0))
quadrocopter_1.set_velocity(Vec3(0, 0, 0))

quadrocopter_1.set_attitude(Rotation.identity())
#start at equilibrium rates:
quadrocopter_1._omega = Vec3(0,0,0)


desPos_2 = Vec3(0, 0, 1.5)

quadrocopter_2.set_position(Vec3(2, 2, 0))
quadrocopter_2.set_velocity(Vec3(0, 0, 0))

quadrocopter_2.set_attitude(Rotation.identity())
#start at equilibrium rates:
quadrocopter_2._omega = Vec3(0,0,0)

#==============================================================================
# Set UAV No.1 and No.2
#==============================================================================

#==============================================================================
# Run the simulation
#==============================================================================

numSteps = int((endTime)/dt)
index = 0

t = 0

posHistory_1, posHistory_2              = np.zeros([numSteps,3]), np.zeros([numSteps,3])
velHistory_1, velHistory_2              = np.zeros([numSteps,3]), np.zeros([numSteps,3])
angVelHistory_1, angVelHistory_2        = np.zeros([numSteps,3]), np.zeros([numSteps,3])
attHistory_1, attHistory_2              = np.zeros([numSteps,3]), np.zeros([numSteps,3])
motForcesHistory_1, motForcesHistory_2  = np.zeros([numSteps,quadrocopter_1.get_num_motors()]), np.zeros([numSteps,quadrocopter_2.get_num_motors()])
inputHistory_1, inputHistory_2          = np.zeros([numSteps,quadrocopter_1.get_num_motors()]), np.zeros([numSteps,quadrocopter_2.get_num_motors()])
times                                   = np.zeros([numSteps,1])

while index < numSteps:
    #define commands:
    accDes_1 = posControl.get_acceleration_command(desPos_1, quadrocopter_1._pos, quadrocopter_1._vel)
    
    if disablePositionControl:
        accDes_1 *= 0 #disable position control
    
    accDes_2 = posControl.get_acceleration_command(desPos_2, quadrocopter_2._pos, quadrocopter_2._vel)
    
    if disablePositionControl:
        accDes_2 *= 0 #disable position control
        
    #==================================== Original version =======================================
    
    #mass-normalised thrust:
    thrustNormDes_1 = accDes_1 + Vec3(0, 0, 9.81)
    angAccDes_1 = attController.get_angular_acceleration(thrustNormDes_1, quadrocopter_1._att, quadrocopter_1._omega)
    motCmds_1 = mixer_1.get_motor_force_cmd(thrustNormDes_1, angAccDes_1)
    
    thrustNormDes_2 = accDes_2 + Vec3(0, 0, 9.81)
    angAccDes_2 = attController.get_angular_acceleration(thrustNormDes_2, quadrocopter_2._att, quadrocopter_2._omega)
    motCmds_2 = mixer_2.get_motor_force_cmd(thrustNormDes_2, angAccDes_2)
    
    #compute the downwash
    quadAbsPos_1, quadAbsPos_2 = quadrocopter_1._pos, quadrocopter_2._pos
    dwForce4Quad_1, dwForce4Quad_2 = Vec3(0, 0, 0), Vec3(0, 0, 0)
    if quadAbsPos_1.z >= quadAbsPos_2.z:
        dwForce4Quad_2 = downwash(quadAbsPos_1, quadAbsPos_2, quadrocopter_1.get_motor_forces())
    else:
        dwForce4Quad_1 = downwash(quadAbsPos_2, quadAbsPos_1, quadrocopter_2.get_motor_forces())
    #run the simulator
    quadrocopter_1.run(dt, motCmds_1) #, dwForce4Quad_1)
    quadrocopter_2.run(dt, motCmds_2) #, dwForce4Quad_2)
    
    #===================================== RL Control ============================================
    # #mass-normalised thrust:
    # thrustNormDes = accDes + Vec3(0, 0, 9.81)
    # #desired ang velocity 
    # angVelDes = attController.get_angular_velocity(thrustNormDes, quadrocopter._att, quadrocopter._omega)
    # cur_state.att = quadrocopter._att.to_array().flatten()
    # cur_state.omega = quadrocopter._omega.to_array().flatten()
    # cur_state.proper_acc = quadrocopter._accel.to_array().flatten()
    # cur_state.cmd_collective_thrust = thrustNormDes.z 
    # cur_state.cmd_bodyrates = angVelDes.to_array().flatten()
    
    # motCmds = low_level_controller.run(cur_state)
    # #run the simulator 
    # quadrocopter.run(dt, motCmds,spdCmd=True)
    #===================================== RL Control ============================================

    #for plotting
    times[index] = t
    inputHistory_1[index,:], inputHistory_2[index,:]            = motCmds_1, motCmds_2 #motForceCmds
    posHistory_1[index,:], posHistory_2[index,:]                = quadrocopter_1._pos.to_list(), quadrocopter_2._pos.to_list()
    velHistory_1[index,:], velHistory_2[index,:]                = quadrocopter_1._vel.to_list(), quadrocopter_2._pos.to_list()
    attHistory_1[index,:], attHistory_2[index,:]                = quadrocopter_1._att.to_euler_YPR(), quadrocopter_2._att.to_euler_YPR()
    angVelHistory_1[index,:], angVelHistory_2[index,:]          = quadrocopter_1._omega.to_list(), quadrocopter_2._omega.to_list()
    motForcesHistory_1[index,:], motForcesHistory_2[index,:]    = quadrocopter_1.get_motor_forces(), quadrocopter_2.get_motor_forces()

    t += dt
    index += 1

#==============================================================================
# Make the plots
#==============================================================================
   
fig, ax = plt.subplots(5,1, sharex=True)

ax[0].plot(times, posHistory_1[:,0], label='x')
ax[0].plot(times, posHistory_1[:,1], label='y')
ax[0].plot(times, posHistory_1[:,2], label='z')
ax[0].plot([0, endTime], [desPos_1.to_list(), desPos_1.to_list()],':')
ax[1].plot(times, velHistory_1)
ax[2].plot(times, attHistory_1[:,0]*180/np.pi, label='Y')
ax[2].plot(times, attHistory_1[:,1]*180/np.pi, label='P')
ax[2].plot(times, attHistory_1[:,2]*180/np.pi, label='R')
ax[3].plot(times, angVelHistory_1[:,0], label='p')
ax[3].plot(times, angVelHistory_1[:,1], label='q')
ax[3].plot(times, angVelHistory_1[:,2], label='r')
ax[4].plot(times, inputHistory_1)
ax[4].plot(times, inputHistory_1)
ax[4].plot(times, motForcesHistory_1,':')

ax[-1].set_xlabel('Time [s]')

ax[0].set_ylabel('Pos')
ax[1].set_ylabel('Vel')
ax[2].set_ylabel('Att [deg]')
ax[3].set_ylabel('AngVel (in B)')
ax[4].set_ylabel('MotForces')

ax[0].set_xlim([0, endTime])
ax[0].legend()
ax[2].legend()
ax[3].legend()

print('Ang vel: ',angVelHistory_1[-1,:])
print('Motor speeds: ',quadrocopter_1.get_motor_speeds())
plt.show()

data_dict = {
    'Time': times.flatten(),
    'PosX_1': posHistory_1[:, 0],
    'PosY_1': posHistory_1[:, 1],
    'PosZ_1': posHistory_1[:, 2],
    'VelX_1': velHistory_1[:, 0],
    'VelY_1': velHistory_1[:, 1],
    'VelZ_1': velHistory_1[:, 2],
    'Yaw_1': attHistory_1[:, 0],
    'Pitch_1': attHistory_1[:, 1],
    'Roll_1': attHistory_1[:, 2],
    'PosX_2': posHistory_2[:, 0],
    'PosY_2': posHistory_2[:, 1],
    'PosZ_2': posHistory_2[:, 2],
    'VelX_2': velHistory_2[:, 0],
    'VelY_2': velHistory_2[:, 1],
    'VelZ_2': velHistory_2[:, 2],
    'Yaw_2': attHistory_2[:, 0],
    'Pitch_2': attHistory_2[:, 1],
    'Roll_2': attHistory_2[:, 2],
}
# Create a DataFrame from the dictionary
data = pd.DataFrame(data_dict)

# Save the DataFrame to a CSV file
data.to_csv('data/multidrones_data.csv', index=False)


# Load your data 
data = pd.read_csv('data/multidrones_data.csv')
times = data['Time'].values

x_list, R_list = [], []

posHistory_1, posHistory_2 = data[['PosX_1', 'PosY_1', 'PosZ_1']].values, data[['PosX_2', 'PosY_2', 'PosZ_2']].values
attHistory_1, attHistory_2 = data[['Yaw_1','Pitch_1', 'Roll_1']].values, data[['Yaw_2','Pitch_2', 'Roll_2']].values
x_1, x_2 = posHistory_1.T, posHistory_2.T
steps = len(times)

R_1, R_2 = np.zeros((3, 3, steps)), np.zeros((3, 3, steps))
for i in range(steps):
    ypr_1, ypr_2 = attHistory_1[i,:], attHistory_2[i,:]
    R_1[:, :, i] = ypr_to_R(ypr_1, degrees=False)
    R_2[:, :, i] = ypr_to_R(ypr_2, degrees=False)
    
x_list = [x_1, x_2]
R_list = [R_1, R_2]

animate_multidrones_history(times, x_list, R_list, [0.4, 0.2], gif_path='data/multi_drones_animation.gif')       
    