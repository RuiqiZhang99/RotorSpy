from __future__ import division, print_function

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import numpy as np

class QuadAttiControllerNested:
    def __init__(self, timeConstantAngleRollPitch, timeConstantAngleYaw, timeConstantRateRollPitch, timeConstantRateYaw):
        #A simple, nested controller.
        self._timeConstAngle_RP = timeConstantAngleRollPitch
        self._timeConstAngle_Y  = timeConstantAngleYaw
        self._timeConstRate_RP = timeConstantRateRollPitch
        self._timeConstRate_Y  = timeConstantRateYaw
        return

    def get_angular_acceleration(self, desNormThrust, curAtt, curAngVel):
        #Step 1: compute desired rates:
        # 1.1: construct a desired attitude, that matches the desired thrust direction
        desThrustDir = desNormThrust / desNormThrust.norm2()

        e3 = Vec3(0,0,1)
        angle = np.arccos(desThrustDir.dot(e3))
        rotAx = e3.cross(desThrustDir)
        n = rotAx.norm2()
        if n < 1e-6:
            #too small to care:
            desAtt = Rotation.identity()
        else:
            desAtt = Rotation.from_rotation_vector(rotAx*(angle/n))
            
        # 1.2 Compute desired rates:
        desRotVec = (desAtt*curAtt.inverse()).to_rotation_vector()

        desAngVel = Vec3(0,0,0)
        desAngVel.x = desRotVec.x/self._timeConstAngle_RP
        desAngVel.y = desRotVec.y/self._timeConstAngle_RP
        desAngVel.z = desRotVec.z/self._timeConstAngle_Y
        
        #Step 2: run the rates controller:
        # 2.1: Compute desired angular acceleration
        desAngAcc = desAngVel - curAngVel
        desAngAcc.x /= self._timeConstRate_RP
        desAngAcc.y /= self._timeConstRate_RP
        desAngAcc.z /= self._timeConstRate_Y
        
        return desAngAcc
    
    def get_angular_velocity(self, desNormThrust, curAtt, curAngVel):
        #Step 1: compute desired rates:
        # 1.1: construct a desired attitude, that matches the desired thrust direction
        desThrustDir = desNormThrust / desNormThrust.norm2()

        e3 = Vec3(0,0,1)
        angle = np.arccos(desThrustDir.dot(e3))
        rotAx = e3.cross(desThrustDir)
        n = rotAx.norm2()
        if n < 1e-6:
            #too small to care:
            desAtt = Rotation.identity()
        else:
            desAtt = Rotation.from_rotation_vector(rotAx*(angle/n))
            
        # 1.2 Compute desired rates:
        desRotVec = (desAtt*curAtt.inverse()).to_rotation_vector()

        desAngVel = Vec3(0,0,0)
        desAngVel.x = desRotVec.x/self._timeConstAngle_RP
        desAngVel.y = desRotVec.y/self._timeConstAngle_RP
        desAngVel.z = desRotVec.z/self._timeConstAngle_Y

        
        return desAngVel
        
class QuadAttiController:
    def __init__(self, time_const_xy=0.064, time_const_z=0.32):
        self.time_const_xy = time_const_xy
        self.time_const_z = time_const_z
        if self.time_const_z < self.time_const_xy:
            # Ensure yaw control is not more aggressive than tilt control
            self.time_const_z = self.time_const_xy
            raise ValueError("Yaw time constant cannot be less than roll/pitch time constant.")

    def get_angular_acceleration(self, curAtt, curAngVel, desAtt=Rotation(1, 0, 0, 0)):
        # Calculate error attitude
        err_att = desAtt.inverse() * curAtt
        des_rot_vec = err_att.to_rotation_vector()

        err_att_inv = err_att.inverse()
        vector_up = Vec3(0, 0, 1)
        des_red_att_rot_ax = (err_att_inv * vector_up).cross(vector_up)
        des_red_att_rot_an_cos = (err_att_inv * vector_up).dot(vector_up)

        if des_red_att_rot_an_cos >= 1.0:
            des_red_att_rot_an = 0
        elif des_red_att_rot_an_cos <= -1.0:
            des_red_att_rot_an = np.pi
        else:
            des_red_att_rot_an = np.arccos(des_red_att_rot_an_cos)

        # Normalize the rotation axis
        n = des_red_att_rot_ax.norm2()
        if n < 1e-12:
            des_red_att_rot_ax = Vec3(0, 0, 0)
        else:
            des_red_att_rot_ax /= n

        k3 = 1.0 / self.time_const_z
        k12 = 1.0 / self.time_const_xy

        desAngVel = -k3 * des_rot_vec - (k12 - k3) * des_red_att_rot_an * des_red_att_rot_ax

        desAngAcc = desAngVel - curAngVel
        desAngAcc.x /= self.time_const_xy
        desAngAcc.y /= self.time_const_xy
        desAngAcc.z /= self.time_const_z
        
        return desAngAcc