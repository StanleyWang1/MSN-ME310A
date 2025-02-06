import depthai as dai

# List all connected DepthAI devices
devices_info = dai.Device.getAllAvailableDevices()

if len(devices_info) == 0:
    print("No DepthAI devices found.")
    exit()

print(f"Found {len(devices_info)} device(s).")

# Iterate through each connected device
for device_info in devices_info:
    print(f"\nGetting calibration data for device: {device_info.getMxId()}")

    # Connect to the device
    with dai.Device() as device:
        # Load calibration data
        calib_data = device.readCalibration()

        # Get intrinsics for the RGB camera
        rgb_intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB)

        # Display intrinsic matrix
        print(f"RGB Camera Intrinsics for device {device_info.getMxId()}:")
        for row in rgb_intrinsics:
            print(row)

        # Get camera-specific parameters
        focal_length = rgb_intrinsics[0][0], rgb_intrinsics[1][1]
        principal_point = rgb_intrinsics[0][2], rgb_intrinsics[1][2]
        print(f"Focal Length: {focal_length}")
        print(f"Principal Point: {principal_point}")

        # Get distortion coefficients
        distortion = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
        print(f"Distortion Coefficients: {distortion}")
