import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Specify the desired ArUco dictionary
desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"

# Define ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

def plot_distances(distances, times):
    plt.figure()
    plt.plot(times, distances, marker='o')
    plt.title("Euclidean Distance vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (pixels)")
    plt.grid(True)
    plt.show()

def main():
    if desired_aruco_dictionary not in ARUCO_DICT:
        print(f"ArUco dictionary '{desired_aruco_dictionary}' not supported.")
        return

    print(f"Detecting '{desired_aruco_dictionary}' markers...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    aruco_params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(1)

    recording = False
    positions = []
    start_time = None
    record_duration = 10  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                corners = marker_corners.reshape((4, 2)).astype(int)
                top_left, top_right, bottom_right, bottom_left = corners

                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

                center_x, center_y = (top_left + bottom_right) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                cv2.putText(frame, str(marker_id), (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                origin_x, origin_y = top_left
                cv2.putText(frame, f"({origin_x}, {origin_y})", (origin_x, origin_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if recording:
                    if start_time is None:
                        start_time = time.time()
                    elapsed_time = time.time() - start_time
                    if elapsed_time <= record_duration:
                        positions.append((origin_x, origin_y))
                    else:
                        recording = False
                        print("Recording complete. Processing data...")

        if recording:
            cv2.putText(frame, "Recording...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('ArUco Marker Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') and not recording:
            recording = True
            positions = []
            start_time = None
            print("Recording started.")

    cap.release()
    cv2.destroyAllWindows()

    if positions:
        distances = [np.linalg.norm(np.array(positions[i]) - np.array(positions[0]))
                     for i in range(1, len(positions))]
        times = np.linspace(0, record_duration, len(distances))
        plot_distances(distances, times)
    else:
        print("No data recorded.")

if __name__ == "__main__":
    main()