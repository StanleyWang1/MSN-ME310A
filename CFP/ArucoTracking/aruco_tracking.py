import cv2
import time
import numpy as np

# Specify the desired ArUco dictionary
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL

def calculate_angle(point1, point2, point3):
    """Calculate the angle (in degrees) between two lines:
    line1: point1 -> point2
    line2: point1 -> point3
    """
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / (norm1 * norm2))
    return np.degrees(angle_rad)

def main():
    # Load the predefined dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Start the webcam
    cap = cv2.VideoCapture(1)

    # Set resolution to 720p (1280x720) and framerate to 60 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify settings
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {int(width)}x{int(height)}, FPS: {fps}")
    
    # Variables to store marker positions
    marker_positions = {1: None, 22: None, 33: None}
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers in the frame
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                corners = marker_corners.reshape((4, 2)).astype(int)
                top_left = tuple(corners[0])
                top_right = tuple(corners[1])

                # Update marker positions
                if marker_id in marker_positions:
                    marker_positions[marker_id] = {"top_left": top_left, "top_right": top_right}

                # Draw marker outline and label
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {marker_id}", top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw lines and calculate angles if all markers are detected
            if all(marker_positions.values()):
                # Get positions
                point1 = marker_positions[1]["top_left"]
                point2 = marker_positions[22]["top_left"]
                point3 = marker_positions[33]["top_right"]

                # Draw lines
                cv2.line(frame, point1, point2, (255, 0, 0), 2)  # Blue line
                cv2.line(frame, point1, point3, (0, 255, 0), 2)  # Green line

                # Calculate angle between the two lines
                angle_between_lines = calculate_angle(point1, point2, point3)

                # Calculate angle between vertical and the 1-33 line
                vertical_point = (point1[0], point1[1] - 100)  # A vertical line from point1
                angle_with_vertical = calculate_angle(point1, vertical_point, point3)

                # Draw the vertical line
                cv2.line(frame, point1, vertical_point, (0, 0, 255), 2)  # Red line

                # Display angles on the frame
                cv2.putText(frame, f"Angle 1-22/1-33: {angle_between_lines:.2f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Angle 1-33/Vertical: {angle_with_vertical:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:  # Update every second
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('ArUco Marker Detector', frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()