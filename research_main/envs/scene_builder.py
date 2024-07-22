import numpy as np
from scipy.spatial.transform import Rotation as R

def create_marker_body(marker_position):
    """
    Creates the XML string for a marker body.

    Parameters:
    marker_position (list or tuple of float): Position [x, y, z] of the marker.

    Returns:
    str: XML string defining the marker body.
    """
    return f'''
    <body name="marker" pos="{marker_position[0]} {marker_position[1]} {marker_position[2]}">
        <geom size="0.01 0.01 0.01" type="sphere" rgba="1 0 0 1" />
    </body>'''


def create_block_bodies(block_positions_orientations):
    """
    Creates the XML string for block bodies.

    Parameters:
    block_positions_orientations (list of tuples): Each tuple contains position [x, y, z] and Euler angles [roll, pitch, yaw].

    Returns:
    str: XML string defining the block bodies.
    """
    block_bodies = ""
    for i, (pos, euler) in enumerate(block_positions_orientations):
        # Convert Euler angles to quaternion
        quat = R.from_euler('zyx', euler).as_quat()
        block_bodies += f'''
        <body name="block_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}">
            <freejoint name="block_joint"/>
            <geom name="orange_subbox" size="0.03 0.01 0.08" pos="0 0.01 0" type="box" rgba="0.8 0.5 0.3 1" friction="4 0.001 0.0001" density="1000"/>
            <geom name="blue_subbox"size="0.03 0.01 0.08" pos="0 -0.01 0" type="box" rgba="0.3 0.5 0.8 1" friction="4 0.001 0.0001" density="1000"/>
        </body>'''
    return block_bodies

def create_ur_model(marker_position=None, block_positions_orientations=None):
    """
    Creates an XML string for a MuJoCo scene with a UR10e robot, an optional marker, and optional blocks.

    Parameters:
    marker_position (list or tuple of float, optional): Position [x, y, z] of the marker.
    block_positions_orientations (list of tuples, optional): Each tuple contains position [x, y, z] and Euler angles [roll, pitch, yaw].

    Returns:
    str: XML string defining the MuJoCo scene.
    """
    marker_body = ""
    if marker_position is not None:
        marker_body = create_marker_body(marker_position)

    block_bodies = ""
    if block_positions_orientations is not None:
        block_bodies = create_block_bodies(block_positions_orientations)

    xml_string = f'''
    <mujoco model="ur10e scene">
        <include file="universal_robots_ur10e_2f85/ur10e_2f85.xml"/>

        <visual>
            <global offheight="2160" offwidth="3840"/>
            <quality offsamples="8"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

            {marker_body}
            {block_bodies}
        </worldbody>
    </mujoco>'''
    
    return xml_string

        

    