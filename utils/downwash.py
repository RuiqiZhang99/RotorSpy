"""
    Copyright @ UC Berkeley HiPeR Lab
    Including the downwash calculation and thrust change on propellers
    References:
    https://arxiv.org/pdf/2207.09645.pdf
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

"""
import numpy as np
from py3dmath import Vec3
import math

'''
airDensity = 1.225
caxDefault = 4.90
cradDefault = 25.00
cdDefault = 1.18

uavPosArr = np.array([
    [1, 1.5, 6],
    [1, 1, 5],
])

uavRpyArr = np.array([
    [0, 0, 0],
    [0, 0, 3.1415926536*0.25],
])

uavInfoArr = np.array([
    [1, 1, 1, 1, 1, 0.1, 1, 3.7e-7],
    [2, 2, 2, 2, 1, 0.2, 1, 3.7e-7],
])
'''

def downwash(upperPosArr, lowerPosArr, upperThrust, VeloCoef=0.5, DragCoef=0.7, z0=0.005):
    '''
        Only for 2 Quadrotors:
        Note: Additional System Identification is required for "uavInfoArr", which should be set up in drone configs 
        under path: uav_sim/configs/multi_drone.yaml
    '''
    assert upperPosArr[2] >= lowerPosArr[2]
    relPos = upperPosArr - lowerPosArr
    radDisp = Vec3(relPos.x, relPos.y, 0).norm2()
    relZPos = relPos.z
    uavThrust = float(np.sum(upperThrust))
    # uavDiscArea = np.pi * np.square(uavInfoArr_ranked[:, 5])
    
    if relZPos <= z0:
        dwForce = 0
    else:
        dwVmax = math.sqrt(uavThrust) / relZPos
        # dwVmax = relZPos
        dwV = dwVmax * math.exp(-(radDisp / relZPos)**2) * VeloCoef
        dwForce = -dwV**2 * DragCoef
    return Vec3(0, 0, dwForce)
'''
    idx = np.argsort(uavPosArr[:, -1]) # Determine the relative Z position, aerodynamics works oneway.
    uavPosArr_ranked = uavPosArr[idx][::-1] 
    # uavInfoArr_ranked = uavInfoArr[idx][::-1]
    
    relPos = uavPosArr_ranked - np.append(np.delete(uavPosArr_ranked, 0, axis=0), np.zeros((1, 3)), axis=0)
    relPos = np.delete(relPos, -1, axis=0)
    radDisp = np.linalg.norm(relPos[:, :-1], axis=1)
    relZPos = relPos[:, -1]
    uavThrust = np.sum(uavThrust, axis=1)
    # uavDiscArea = np.pi * np.square(uavInfoArr_ranked[:, 5])
    dwVmax = np.sqrt(uavThrust)[:-1] / relZPos
    
    # dwVmax = relZPos
    dwV = dwVmax * np.exp(-np.square(radDisp / relZPos)) * VeloCoef
    dwForce = -np.square(dwV) * DragCoef
    return dwForce
'''

'''
def thrust_change(uavPosArr, uavRpyArr, uavInfoArr):
    idx = np.argsort(uavPosArr[:, -1])
    uavPosArr, uavRpyArr, uavInfoArr = uavPosArr[idx][::-1], uavRpyArr[idx][::-1], uavInfoArr[idx][::-1]
    yaw = uavRpyArr[-1, -1]
    
    relPos = uavPosArr - np.append(np.delete(uavPosArr, 0, axis=0), np.zeros((1, 3)), axis=0)
    relPos = np.delete(relPos, -1, axis=0)
    relZPos = relPos[:, -1]
    uavBv = uavInfoArr[-1, 7]
    uavThrust = np.sum(uavInfoArr[:, :4], axis=1)
    uavDiscArea = np.pi * np.square(uavInfoArr[:, 5])
    dwVmax = np.sqrt(uavThrust / (2.45 * uavDiscArea))[:-1] * caxDefault / relZPos
    
    rotMatrix = np.array([
        [np.cos(yaw), -np.sin(yaw)], 
        [np.sin(yaw), np.cos(yaw)]])
    armLength = uavInfoArr[-1, 4]
    motor_pos = armLength/(2**0.5)
    
    motor0RelPos =  relPos[:, :-1] + np.dot(rotMatrix, np.array([+motor_pos, -motor_pos]).T)
    motor1RelPos =  relPos[:, :-1] + np.dot(rotMatrix, np.array([-motor_pos, -motor_pos]).T)
    motor2RelPos =  relPos[:, :-1] + np.dot(rotMatrix, np.array([-motor_pos, +motor_pos]).T)
    motor3RelPos =  relPos[:, :-1] + np.dot(rotMatrix, np.array([+motor_pos, +motor_pos]).T)
    
    motor0Factor = 1 - uavBv * dwVmax * np.exp(-cradDefault * np.sum(np.square(motor0RelPos)) / np.square(relZPos))
    motor1Factor = 1 - uavBv * dwVmax * np.exp(-cradDefault * np.sum(np.square(motor1RelPos)) / np.square(relZPos))
    motor2Factor = 1 - uavBv * dwVmax * np.exp(-cradDefault * np.sum(np.square(motor2RelPos)) / np.square(relZPos))
    motor3Factor = 1 - uavBv * dwVmax * np.exp(-cradDefault * np.sum(np.square(motor3RelPos)) / np.square(relZPos))
    
    return motor0Factor, motor1Factor, motor2Factor, motor3Factor
'''
# Our quadcopter model as    
#           
#           x
#           ^
#   (-)mot3 | mot0(+)
#           |
#     y<----+-----
#           |
#   (+)mot2 | mot1(-)   

import math

def downwash_karan(relativePos):

    centerLineForce = 6.8  # [m/s^2]
    kForce = -20.8  # [1/m^2]
    maxExtraForce = 2.0  # [m/s^2]
    dockSep = 0.05  

    # Distance on X-Y axes
    radialSepSq = relativePos.x**2 + relativePos.y**2
    extraForce = 0.0

    # Calculate the Downwash Force
    if relativePos.z <= dockSep + 0.2:
        extraForce = (1 - (relativePos[2] - dockSep) / 0.2) * maxExtraForce

    # Too Far
    if radialSepSq > 0.5**2 or relativePos.z < dockSep:
        return Vec3(0, 0, 0)
    else:
        downwashForce = (centerLineForce + extraForce) * math.exp(kForce * radialSepSq)
        return Vec3(0, 0, -downwashForce) 