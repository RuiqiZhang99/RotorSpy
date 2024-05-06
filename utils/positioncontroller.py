from __future__ import print_function, division

class PositionController:
    """A simple position controller, with acceleration as output. 
    
    Makes position behave as a second order system. 
    """
    
    def __init__(self, natFreq=3.8, dampingRatio=0.7):
        self._natFreq = natFreq
        self._dampingRatio = dampingRatio
        
        
    def get_acceleration_command(self, desPos, curPos, curVel):
        return self._natFreq**2*(desPos - curPos) - 2 * self._dampingRatio * self._natFreq * curVel 
