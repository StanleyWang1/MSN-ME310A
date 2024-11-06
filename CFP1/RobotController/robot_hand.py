import threading
import time
from dynamixel_controller import DynamixelController
import numpy as np
import cv2
import mediapipe as mp

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
    alphaOne = np.arccos(L13**2 / (2*L_link*L13))
    betaOne = np.arctan2(y, -x)
    thetaOne = np.pi - alphaOne - betaOne
    alphaFive = np.arctan2(y, x+L_base)
    betaFive = np.arccos((L53**2) / (2*L53*L_link))
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
profile_velocity_limit = 50
controller.WRITE(LEFT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(RIGHT_BASE_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 
controller.WRITE(WRIST_MOTOR, PROFILE_VELOCITY, profile_velocity_limit) 

# -------------------- HAND TRACKING STUFF --------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#pixel locations
[pixel_x1, pixel_y1] = [926, 26]  # top left
[pixel_x2, pixel_y2] = [1700, 24]  # top right
[pixel_x3, pixel_y3] = [929, 1042]  # bottom right
[pixel_x4, pixel_y4] = [1717, 1037]   # bottom left

src_points = np.float32([
    [pixel_x1, pixel_y1],
    [pixel_x2, pixel_y2],
    [pixel_x3, pixel_y3],
    [pixel_x4, pixel_y4]
])

# real world points
dst_points = np.float32([
    [0, 0],      # top left
    [0.2159, 0],     # top right
    [0, 0.2794],    # bottom right
    [0.2159, 0.2794]      # bottom left
])

transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

def map_pixel_to_real(pixel_coords):
    pixel_coords = np.array([[pixel_coords]], dtype='float32')
    real_world_coords = cv2.perspectiveTransform(pixel_coords, transformation_matrix)
    return tuple(np.round(real_world_coords[0][0], 2))  # Round to 2 decimal places

def calculate_angle(vec1, vec2):
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

# Robot kinematics
x = 0
y = 0.15
wrist_offset = 0
gripper_pwm = 0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                h, w, _ = frame.shape

                thumb_tip = hand_landmarks.landmark[4]
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_base = hand_landmarks.landmark[1]

                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                thumb_base_coords = (int(thumb_base.x * w), int(thumb_base.y * h))

                thumb_tip_real = map_pixel_to_real(thumb_tip_coords)
                index_finger_tip_real = map_pixel_to_real(index_finger_tip_coords)
                thumb_base_real = map_pixel_to_real(thumb_base_coords)

                vec_thumb = np.array(thumb_tip_coords) - np.array(thumb_base_coords)
                vec_index = np.array(index_finger_tip_coords) - np.array(thumb_base_coords)
                vec_vertical = np.array([0, -1])  # vertical line pointing up

                angle_between_lines = calculate_angle(vec_thumb, vec_index)
                angle_with_vertical = calculate_angle(vec_thumb, vec_vertical)
                if thumb_base_real[0] > thumb_tip_real[0]:
                    angle_with_vertical = -1 * angle_with_vertical
                
                cv2.putText(frame, f"Thumb Tip: {thumb_tip_real}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Index Tip: {index_finger_tip_real}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Thumb Base: {thumb_base_real}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pincher Angle: {angle_between_lines:.2f}deg", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Thumb Angle: {angle_with_vertical:.2f}deg", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.line(frame, thumb_tip_coords, thumb_base_coords, (255, 0, 0), 2)  # blue line for thumb tip to base
                cv2.line(frame, thumb_base_coords, index_finger_tip_coords, (0, 0, 255), 2)  # red line for thumb base to index tip

                cv2.circle(frame, thumb_tip_coords, 5, (0, 255, 0), -1)
                cv2.circle(frame, index_finger_tip_coords, 5, (0, 255, 0), -1)
                cv2.circle(frame, thumb_base_coords, 5, (0, 255, 0), -1)

                # Robot calculation
                x_hand = thumb_base_real[0]
                y_hand = thumb_base_real[1]
                x_robot = x_hand/2
                y_robot = 0.25 - y_hand/2
                # Calculate IK based on robot coordinates
                ticks1, ticks2, ticks3 = IK(x_robot, y_robot)
                wrist_offset = int(radians_to_ticks(angle_with_vertical / 180 * np.pi))
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
                
        cv2.imshow('Hand Tracking with Real-World Coordinates and Angles', frame)

        # exit on pressing q
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

cap.release()
cv2.destroyAllWindows()


# try:
#     while not stop_flag:
#         ticks1, ticks2, ticks3 = IK(x, y)

#         # Update motor positions
#         controller.WRITE(LEFT_BASE_MOTOR, GOAL_POSITION, ticks1)
#         controller.WRITE(RIGHT_BASE_MOTOR, GOAL_POSITION, ticks2)
#         controller.WRITE(WRIST_MOTOR, GOAL_POSITION, ticks3 + wrist_offset)
#         controller.WRITE(GRIPPER_MOTOR, GOAL_PWM, gripper_pwm)

#         # Read actual positions
#         ticks1_real = controller.READ(LEFT_BASE_MOTOR, PRESENT_POSITION)
#         ticks2_real = controller.READ(RIGHT_BASE_MOTOR, PRESENT_POSITION)
#         ticks3_real = controller.READ(WRIST_MOTOR, PRESENT_POSITION)

#         # Print position error
#         print(f"e1: {ticks1_real - ticks1}, e2: {ticks2_real - ticks2}, e3: {ticks3_real - ticks3}")

#         time.sleep(1/100)

# finally:
#     listener.stop()
