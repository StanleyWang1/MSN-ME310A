import mujoco
import numpy as np
import time
from mujoco.viewer import launch

def simulate_sphere(x: float, y: float, z: float):
    """
    Simulates a sphere following a sinusoidal trajectory along the x-axis.

    Parameters:
    - y (float): Constant y-position in mm.
    - z (float): Constant z-position in mm.
    - amplitude (float): Amplitude of the sinusoidal motion (meters).
    - frequency (float): Frequency of the sinusoidal motion (Hz).
    - duration (float): Duration of the simulation (seconds).
    """

    # Convert positions from mm to meters
    x, y, z = x / 1000.0, y / 1000.0, z / 1000.0

    # Define an XML model string for the ball simulation
    model_xml = f"""
    <mujoco model="sinusoidal_sphere_simulation">
        <worldbody>
            <!-- Ground plane -->
            <geom name="ground" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
            
            <!-- Sphere object with a free joint -->
            <body name="ball" pos="0 {y} {z}">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.03" rgba="1 0 0 1"/>
            </body>
        </worldbody>
        <option timestep="0.01"/>
    </mujoco>
    """

    # Create the model and simulation
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)

    # Initialize simulation parameters
    start_time = time.time()
    current_time = 0.0
    timestep = model.opt.timestep

    while current_time < duration:
        # Calculate the sinusoidal x-position based on current time
        x_pos = amplitude * np.sin(2 * np.pi * frequency * current_time)

        # Update the position of the ball (qpos has 7 elements for freejoint: [x, y, z, qx, qy, qz, qw])
        data.qpos[:3] = [x_pos, y, z]

        # Step the simulation
        mujoco.mj_step(model, data)

        # Update time and render
        current_time = time.time() - start_time
        mujoco.mj_forward(model, data)

        # Sleep to match real-time simulation speed
        time.sleep(timestep)

    # Launch the viewer to visualize the final result (optional)
    launch(model, data)

# Example usage: sinusoidal trajectory with y = 200 mm, z = 300 mm, amplitude 0.1 meters, frequency 0.5 Hz
simulate_sphere_sinusoidal(y=200, z=300, amplitude=0.1, frequency=0.5, duration=10.0)
