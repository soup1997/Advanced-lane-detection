# Linear Kalman Filter

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf

class WindowFilter:
    def __init__(self, pos_init= None, meas_variance=50, process_variance=1, uncertanity_init = 2 ** 30):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.dt = 0.5
        
        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Process noise
        self.kf.P = np.eye(self.kf.dim_x) * uncertanity_init

        # State transition function
        self.kf.F = np.array([[1., self.dt], 
                              [0., 1.]])
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=process_variance)
        
        # Measurement function
        self.kf.H = np.array([[1., 0.]])
        
        # Measurement noise
        self.kf.R = np.array([[meas_variance]])

    def update(self, pos):
        self.kf.predict()
        self.kf.update(pos)
    
    def grow_uncertainty(self, mag):
        for i in range(mag):
            self.kf.P = self.kf._alpha_sq * np.dot(np.dot(self.kf.F, self.kf.P), self.kf.F.T) + self.kf.Q
    
    def loglikelihood(self, pos):
        self.kf.S = np.dot(np.dot(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)
    
    def get_position(self):
        return self.kf.x[0]