#! /usr/bin/env python

from EKF import EKF

class FusionEKF:
    def __init__(self):
        # create EKF object here and initialize it
        ekf_ = EKF()
    
    def process_GPS(self, msg):
        latitude = msg.latitude
        longitude = msg.longitude
        utm_x, utm_y = utm.from_latlon(latitude, longitude)
        

    def process_IMU(self):
        pass