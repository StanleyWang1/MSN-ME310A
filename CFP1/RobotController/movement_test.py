import threading
import time
from dynamixel_controller import DynamixelController
import numpy as np

# XH-430-W250-T Serial Addresses
OPERATING_MODE = (11, 1)
HOMING_OFFSET = (20, 4)
PWM_LIMIT = (36, 2)
TORQUE_ENABLE = (64, 1)
LED_ENABLE = (65, 1)
POSITION_D_GAIN = (80, 2) # 10000
POSITION_I_GAIN = (82, 2) # 1000
POSITION_P_GAIN = (84, 2) # 1000
GOAL_PWM = (100, 2)
GOAL_VELOCITY = (104, 4)
PROFILE_ACCELERATION = (108, 4)
PROFILE_VELOCITY = (112, 4)
GOAL_POSITION = (116, 4)
PRESENT_LOAD = (126, 2)
PRESENT_VELOCITY = (128, 4)
PRESENT_POSITION = (132, 4)

# Motor IDs
LEFT_BASE_MOTOR = 12
RIGHT_BASE_MOTOR = 13
WRIST_MOTOR = 3
GRIPPER_MOTOR = 4

# Kinematic Parameters
L_base = 0.075 # [m]
L_link = 0.150 # [m]
center_x, center_y = 0, 0.2
radius = 0.05
speed = 0.5  # speed in radians per second

# Initialize controller
controller = DynamixelController('/dev/tty.usbserial-FT89FIU6', 57600, 2.0)

# Set Control Mode
controller.WRITE(LEFT_BASE_MOTOR, OPERATING_MODE, 4) # extended position control 
controller.WRITE(RIGHT_BASE_MOTOR, OPERATING_MODE, 4) # extended position control 
controller.WRITE(WRIST_MOTOR, OPERATING_MODE, 3) # position control
controller.WRITE(GRIPPER_MOTOR, OPERATING_MODE, 3) # position control
controller.WRITE(GRIPPER_MOTOR, PWM_LIMIT, 100) # prevent overgrippping

# Gain Tuning
controller.WRITE(LEFT_BASE_MOTOR, POSITION_P_GAIN, 500)
controller.WRITE(LEFT_BASE_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(LEFT_BASE_MOTOR, POSITION_D_GAIN, 50)

controller.WRITE(RIGHT_BASE_MOTOR, POSITION_P_GAIN, 500)
controller.WRITE(RIGHT_BASE_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(RIGHT_BASE_MOTOR, POSITION_D_GAIN, 50)

controller.WRITE(WRIST_MOTOR, POSITION_P_GAIN, 500)
controller.WRITE(WRIST_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(WRIST_MOTOR, POSITION_D_GAIN, 2000)

# Torque Enable
controller.WRITE(LEFT_BASE_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(RIGHT_BASE_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(WRIST_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(GRIPPER_MOTOR, TORQUE_ENABLE, 1)

# Limit Profile Velocity
profile_velocity_limit = 50
controller.WRITE(LEFT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(RIGHT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(WRIST_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 

def ticks_to_radians(ticks):
    return ticks / 4095 * 2 * np.pi

def radians_to_ticks(theta):
    return theta / (2 * np.pi) * 4095

def IK(x, y):
    x = x - L_base / 2
    L13 = np.sqrt(x**2 + y**2)
    L53 = np.sqrt((x + L_base)**2 + y**2)
    alphaOne = np.arccos(L13**2 / (2 * L_link * L13))
    betaOne = np.arctan2(y, -x)
    thetaOne = np.pi - alphaOne - betaOne
    alphaFive = np.arctan2(y, x + L_base)
    betaFive = np.arccos((L53**2) / (2 * L53 * L_link))
    thetaFive = alphaFive + betaFive
    ticks1 = int(radians_to_ticks(thetaFive) - 1020)
    ticks2 = int(radians_to_ticks(thetaOne) + 3090)

    gamma = np.pi / 2 - betaOne + alphaOne
    ticks3 = int(radians_to_ticks(gamma) + 1012)

    return ticks1, ticks2, ticks3

# Control Thread Function
def control_thread():
    start_time = time.time()
    while True:
        # Calculate current angle based on elapsed time
        elapsed_time = time.time() - start_time
        angle = elapsed_time * speed

        # Calculate circular trajectory point
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)

        # Calculate inverse kinematics for the robot
        ticks1, ticks2, ticks3 = IK(x, y)

        # Update motor positions
        controller.WRITE(LEFT_BASE_MOTOR, GOAL_POSITION, ticks1)
        controller.WRITE(RIGHT_BASE_MOTOR, GOAL_POSITION, ticks2)
        controller.WRITE(WRIST_MOTOR, GOAL_POSITION, ticks3)

        # Control loop frequency
        time.sleep(0.005)  # Adjust as needed

# Start control thread
control_t = threading.Thread(target=control_thread)
control_t.start()

control_t.join()
