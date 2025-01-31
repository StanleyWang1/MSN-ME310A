import cv2
import depthai as dai
import numpy as np
import time
import threading

# Shared frame storage for both cameras
frames = [None, None]
fps_values = [0.0, 0.0]

# Camera processing function
def process_camera(device_info, index):
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Camera properties
    camRgb.setPreviewSize(640, 360)  # Adjust preview size
    camRgb.setFps(60)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setPreviewKeepAspectRatio(False)  # Disable aspect ratio enforcement to avoid cropping

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Connect to device
    with dai.Device(pipeline, device_info) as device:
        print(f'Connected to: {device_info.getMxId()}')

        qRgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)

        frame_count = 0
        start_time = time.time()

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()

            # Convert to HSV and apply mask for highlights
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_hsv = np.array([0, 0, 204])
            upper_hsv = np.array([180, 51, 255])
            mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps_values[index] = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Add FPS overlay
            cv2.putText(result, f"FPS: {fps_values[index]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Store the frame for display
            frames[index] = result

# Get connected devices
available_devices = dai.Device.getAllAvailableDevices()
if len(available_devices) < 2:
    print("Error: Less than two OAK-D cameras connected.")
else:
    threads = []
    for i, device_info in enumerate(available_devices[:2]):
        thread = threading.Thread(target=process_camera, args=(device_info, i))
        threads.append(thread)
        thread.start()

    # Display loop
    while True:
        # Check if both frames are available
        if frames[0] is not None and frames[1] is not None:
            # Stack frames horizontally and display them
            combined_display = np.hstack((frames[0], frames[1]))
            cv2.imshow("Dual Camera View", combined_display)

        if cv2.waitKey(1) == ord('q'):
            break

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
