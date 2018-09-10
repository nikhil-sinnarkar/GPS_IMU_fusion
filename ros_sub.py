#! /usr/bin/env python

import rospy
import utm
from nav_msgs.msg import Odometry
from sensor_msgs.msg import imu
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from FusionEKF import FusionEKF

def gps_callback(msg):
   fusionEKF.process_GPS(msg)

def odom_callback(msg):
    pass

def imu_callback(msg):
    fusionEKF.process_IMU(msg)

def hdg_callback(msg):
    pass


rospy.init_node('ros_sub')

# create FusionEKF object here
fusionEKF = FusionEKF()

# create subscriber for differnt sensor topics
odom_sub = rospy.Subscriber('/mavros/global_position/local', Odometry, odom_callback)
imu_sub = rospy.Subscriber('/mavros/imu/data_raw', Odometry, imu_callback)
gps_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, gps_callback)
hdg_sub = rospy.Subscriber('/mavros/global_position/compass_hdg', Float64, hdg_callback)

rospy.spin()