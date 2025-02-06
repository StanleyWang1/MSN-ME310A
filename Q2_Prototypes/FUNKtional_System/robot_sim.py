import mujoco
import time
import numpy as np
from mujoco.viewer import launch_passive

# Global viewer instance
viewer = None
model = None
data = None

def initialize_simulation():
    """
    Initializes the MuJoCo model, data, and viewer. Called only once.
    """
    global viewer, model, data

    # Load the model from XML string (or path to XML file)
    xml_path = "./Q2_Prototypes/FUNKtional_System/robot.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Launch the passive viewer
    viewer = launch_passive(model, data)


def command_sphere_position(point_positions):
    """
    Commands the sphere to specified x, y, z positions using slider joints.
    The viewer is synchronized after each step.
    
    Parameters:
    - x_target (float): Target position along the x-axis (in meters).
    - y_target (float): Target position along the y-axis (in meters).
    - z_target (float): Target position along the z-axis (in meters).
    """

    # Ensure the simulation is initialized
    if viewer is None:
        raise RuntimeError("Simulation has not been initialized. Call `initialize_simulation()` first.")

    # Update control inputs for the actuators
    data.ctrl[0] = point_positions[0][0]/1000.0  # Target position for x-axis
    data.ctrl[1] = point_positions[0][1]/1000.0  # Target position for y-axis
    data.ctrl[2] = point_positions[0][2]/1000.0  # Target position for z-axis

    data.ctrl[3] = point_positions[1][0]/1000.0  # Target position for x-axis
    data.ctrl[4] = point_positions[1][1]/1000.0  # Target position for y-axis
    data.ctrl[5] = point_positions[1][2]/1000.0  # Target position for z-axis

    data.ctrl[6] = point_positions[2][0]/1000.0  # Target position for x-axis
    data.ctrl[7] = point_positions[2][1]/1000.0  # Target position for y-axis
    data.ctrl[8] = point_positions[2][2]/1000.0  # Target position for z-axis
    # Step the simulation and sync the viewer
    mujoco.mj_step(model, data)
    viewer.sync()


def close_simulation():
    """
    Closes the viewer and cleans up the simulation.
    """
    global viewer

    if viewer is not None:
        viewer.close()
        viewer = None