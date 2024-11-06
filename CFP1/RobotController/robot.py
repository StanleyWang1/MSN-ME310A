from dynamixel_controller import DynamixelController
import time
import numpy as np
from pynput import keyboard

# XH-430-W250-T Serial Addresses
OPERATING_MODE = (11, 1)
HOMING_OFFSET = (20, 4)
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
LEFT_BASE_MOTOR = 1
RIGHT_BASE_MOTOR = 2
WRIST_MOTOR = 3
GRIPPER_MOTOR = 4

# Kinematic Parameters
L_base = 0.075 # [m]
L_link = 0.150 # [m]
def ticks_to_radians(ticks):
    return ticks/4095*2*np.pi
def radians_to_ticks(theta):
    return theta/(2*np.pi)*4095
def IK(x, y):
    x = x - L_base/2
    L13 = np.sqrt(x**2 + y**2)
    L53 = np.sqrt((x+L_base)**2 + y**2)
    alphaOne = np.acos(L13**2 / (2*L_link*L13))
    betaOne = np.atan2(y, -x)
    thetaOne = np.pi - alphaOne - betaOne
    alphaFive = np.atan2(y, x+L_base)
    betaFive = np.acos((L53**2) / (2*L53*L_link))
    thetaFive = alphaFive + betaFive
    ticks1 = int(radians_to_ticks(thetaFive) + 2048)
    ticks2 = int(radians_to_ticks(thetaOne) + 1032)

    gamma = np.pi/2 - betaOne  + alphaOne
    ticks3 = int(radians_to_ticks(gamma) + 1012)

    return ticks1, ticks2, ticks3

controller = DynamixelController('/dev/tty.usbserial-FT89FIU6', 57600, 2.0)

# Set Control Mode
controller.WRITE(LEFT_BASE_MOTOR, OPERATING_MODE, 4) # extended position control 
controller.WRITE(RIGHT_BASE_MOTOR, OPERATING_MODE, 4) # extended position control 
controller.WRITE(WRIST_MOTOR, OPERATING_MODE, 3) # position control
controller.WRITE(GRIPPER_MOTOR, OPERATING_MODE, 16) # PWM control

# Gain Tuning (couldn't get it better than dynamixel built in)
controller.WRITE(LEFT_BASE_MOTOR, POSITION_P_GAIN, 1000)
controller.WRITE(LEFT_BASE_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(LEFT_BASE_MOTOR, POSITION_D_GAIN, 3600)

controller.WRITE(RIGHT_BASE_MOTOR, POSITION_P_GAIN, 1000)
controller.WRITE(RIGHT_BASE_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(RIGHT_BASE_MOTOR, POSITION_D_GAIN, 3600)

controller.WRITE(WRIST_MOTOR, POSITION_P_GAIN, 1000)
controller.WRITE(WRIST_MOTOR, POSITION_I_GAIN, 0)
controller.WRITE(WRIST_MOTOR, POSITION_D_GAIN, 3600)
# Torque Enable
controller.WRITE(LEFT_BASE_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(RIGHT_BASE_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(WRIST_MOTOR, TORQUE_ENABLE, 1)
controller.WRITE(GRIPPER_MOTOR, TORQUE_ENABLE, 1)
# Limit Profile Velocity
profile_velocity_limit = 100
controller.WRITE(LEFT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(RIGHT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(WRIST_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 

x = 0
y = 0.15
wrist_offset = 0
gripper_pwm = 0
step_size = 0.005
stop_flag = False

def on_press(key):
    global x, y, wrist_offset, gripper_pwm, stop_flag
    try:
        if key.char == 'w':  # Move up (+y)
            y += step_size
        elif key.char == 's':  # Move down (-y)
            y -= step_size
        elif key.char == 'a':  # Move left (-x)
            x -= step_size
        elif key.char == 'd':  # Move right (+x)
            x += step_size
        elif key.char == 'q':  
            wrist_offset -= 50
        elif key.char == 'e':  
            wrist_offset += 50
        elif key.char == 'g':  
            gripper_pwm = -100
        elif key.char == 'r':  
            gripper_pwm = 100
    except AttributeError:
        # Stop the program if 'esc' is pressed
        if key == keyboard.Key.esc:
            stop_flag = True
            return False

def on_release(key):
    global gripper_pwm
    try:
        # Reset gripper PWM to 0 when 'g' or 'r' is released
        if key.char == 'g' or key.char == 'r':
            gripper_pwm = 0
    except AttributeError:
        pass

# Start the listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while not stop_flag:
        ticks1, ticks2, ticks3 = IK(x, y)

        # Update motor positions
        controller.WRITE(LEFT_BASE_MOTOR, GOAL_POSITION, ticks1)
        controller.WRITE(RIGHT_BASE_MOTOR, GOAL_POSITION, ticks2)
        controller.WRITE(WRIST_MOTOR, GOAL_POSITION, ticks3 + wrist_offset)
        controller.WRITE(GRIPPER_MOTOR, GOAL_PWM, gripper_pwm)

        # Read actual positions
        ticks1_real = controller.READ(LEFT_BASE_MOTOR, PRESENT_POSITION)
        ticks2_real = controller.READ(RIGHT_BASE_MOTOR, PRESENT_POSITION)
        ticks3_real = controller.READ(WRIST_MOTOR, PRESENT_POSITION)

        # Print position error
        print(f"e1: {ticks1_real - ticks1}, e2: {ticks2_real - ticks2}, e3: {ticks3_real - ticks3}")

        time.sleep(1/100)

finally:
    listener.stop()
