import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#pixel locations
[pixel_x1, pixel_y1] = [361, 134]  # top left
[pixel_x2, pixel_y2] = [1493, 134]  # top right
[pixel_x3, pixel_y3] = [1493, 938]  # bottom right
[pixel_x4, pixel_y4] = [361, 943]   # bottom left

src_points = np.float32([
    [pixel_x1, pixel_y1],
    [pixel_x2, pixel_y2],
    [pixel_x3, pixel_y3],
    [pixel_x4, pixel_y4]
])

# real world points
dst_points = np.float32([
    [0, 0],      # top left
    [14, 0],     # top right
    [14, 10],    # bottom right
    [0, 10]      # bottom left
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

        cv2.imshow('Hand Tracking with Real-World Coordinates and Angles', frame)

        # exit on pressing q
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

cap.release()
cv2.destroyAllWindows()