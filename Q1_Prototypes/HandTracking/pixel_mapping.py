import cv2

# Callback function to display coordinates on click
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: ({x}, {y})")
        # Draw a small circle at the clicked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Display the coordinates on the frame
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up a window and bind the mouse click event to the callback function
cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", show_coordinates)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


[pixel_x1, pixel_y1] = [361, 134],  # top left
[pixel_x2, pixel_y2] = [1493, 134],  # top right
[pixel_x3, pixel_y3] = [1493, 938],  # bottom right
[pixel_x4, pixel_y4] = [361, 943]   # bottom left