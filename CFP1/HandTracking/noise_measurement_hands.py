import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

recording = False
positions = []
start_time = None
record_duration = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if recording:
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                if elapsed_time <= record_duration:
                    positions.append((cx, cy))
                else:
                    recording = False
            cv2.putText(frame, f"Index Tip: ({cx}, {cy})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if recording:
        cv2.putText(frame, "Recording...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and not recording:
        recording = True
        positions = []
        start_time = None

cap.release()
cv2.destroyAllWindows()

if positions:
    distances = [np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
                 for i in range(1, len(positions))]
    times = np.linspace(0, record_duration, len(distances))

    plt.figure()
    plt.plot(times, distances)
    plt.title("Positional Error vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (pixels)")
    plt.grid(True)
    plt.show()