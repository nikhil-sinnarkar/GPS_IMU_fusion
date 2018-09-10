#! /usr/bin/env python

# State Vector - Constant Turn Rate and Velocity Vehicle Model (CTRV)
# x = [PositionX, PositionY, Heading, Velocity, Yawrate]

class EKF:
    def __init__(self):
       self.numstates = 5 # States
       self.state = [] * self.numstates
       # Sensor readings frequency
       self.dt = 1.0/50.0 # Sample Rate of the Measurements is 50Hz
       self.dtGPS = 1.0/10.0 # Sample Rate of GPS is 10Hz

       # Initial Uncertainty
       self.P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

       # Process Noise Covariance Matrix
       self.sGPS     = 0.5*8.8*self.dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
       self.sCourse  = 0.1*self.dt # assume 0.1rad/s as maximum turn rate for the vehicle
       self.sVelocity= 8.8*self.dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
       self.sYaw     = 1.0*self.dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
       self.Q = np.diag([self.sGPS**2, self.sGPS**2, self.sCourse**2, self.sVelocity**2, self.sYaw**2])

       # Measurement Noise Covariance R
       self.varGPS = 6.0  # Standard Deviation of GPS Measurement# Stand
       self.varspeed = 1.0 # Variance of the speed measurement
       self.varyaw = 0.1 # Variance of the yawrate measurement
       self.R = np.matrix([[self.varGPS**2, 0.0],
                          [0.0, self.varGPS**2]])
       # Identity Matrix
       self.I = np.eye(self.numstates)

    def predict(self, yawrate, dt):
       # Time Update (Prediction)
       # ========================
       # Project the state ahead
       # see "Dynamic Matrix"
       if np.abs(yawrate)<0.0001: # Driving straight
           self.state[0] = self.state[0] + self.state[3]*dt * np.cos(self.state[2])
           self.state[1] = self.state[1] + self.state[3]*dt * np.sin(self.state[2])
           self.state[2] = self.state[2]
           self.state[3] = self.state[3]
           self.state[4] = 0.0000001 # avoid numerical issues in Jacobians
           # dstate.append(0)
       else: # otherwise
           self.state[0] = self.state[0] + (self.state[3]/self.state[4]) * (np.sin(self.state[4]*dt+self.state[2]) - np.sin(self.state[2]))
           self.state[1] = self.state[1] + (self.state[3]/self.state[4]) * (-np.cos(self.state[4]*dt+self.state[2])+ np.cos(self.state[2]))
           self.state[2] = (self.state[2] + self.state[4]*dt + np.pi) % (2.0*np.pi) - np.pi
           self.state[3] = self.state[3]
           self.state[4] = self.state[4]
           # dstate.append(1)

    def update(self, x, dt, measurements):  # measurements = [UTMx-datumx,UTMy-datumy]
       # Calculate the Jacobian of the Dynamic Matrix A
       # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
       a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
       a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
       a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
       a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
       a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
       a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
       JA = np.matrix([[1.0, 0.0, a13, a14, a15],
                       [0.0, 1.0, a23, a24, a25],
                       [0.0, 0.0, 1.0, 0.0, dt],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0]])


       # Project the error covariance ahead
       self.P = JA*self.P*JA.T + self.Q

       # Measurement Update (Correction)
       # ===============================
       # Measurement Function
       hx = np.matrix([[float(x[0])],
                       [float(x[1])]])

       # if GPS[filterstep]: # with 10Hz, every 5th step
       JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0]])
       # else: # every other step
       #     JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
       #                     [0.0, 0.0, 0.0, 0.0, 0.0]])

       S = JH*self.P*JH.T + self.R
       K = (self.P*JH.T) * np.linalg.inv(S)

       # Update the estimate via
       Z = measurements.reshape(JH.shape[0],1)  #eg. measurement = [0.27999943 0.66795491]
       y = Z - (hx)  # Innovation or Residual
       x = x + (K*y)
       self.state = x

       # Update the error covariance
       self.P = (self.I - (K*JH))*self.P