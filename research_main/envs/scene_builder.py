import os
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


class SimulationRenderer:
    def __init__(self, model, data, output_dir, local_time, render_modes, contact_vis, framerate=60):
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.render_modes = render_modes or []
        self.contact_vis = contact_vis
        self.framerate = framerate
        self.frames = {}
        self.renderers = {}
        self.cameras = {}
        self.outs = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for mode in self.render_modes:
            self.cameras[mode] = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self.cameras[mode])
            
            if mode == 'livecam':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.renderers[mode] = mujoco.Renderer(self.model, height=1024, width=1440)
                self.frames[mode] = []
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.outs[mode] = cv2.VideoWriter(os.path.join(self.output_dir, f'{mode}_{local_time}.mp4'), fourcc, self.framerate, (1440, 1024))

        if self.contact_vis:
            self.options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.options)
            self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            # tweak scales of contact visualization elements
            self.model.vis.scale.contactwidth = 0.1
            self.model.vis.scale.contactheight = 0.03
            self.model.vis.scale.forcewidth = 0.05
            self.model.vis.map.force = 0.3

    def render_frame(self, time):
        if 'livecam' in self.render_modes:
            self.viewer.sync()
        
        for mode in self.render_modes:
            if mode == 'livecam':
                continue

            if len(self.frames[mode]) < time * self.framerate:
                if mode == 'video_top':
                    self.cameras[mode].distance = 3.2
                    self.cameras[mode].azimuth = 180
                    self.cameras[mode].elevation = -50
                elif mode == 'video_side':
                    self.cameras[mode].distance = 3
                    self.cameras[mode].azimuth = 130
                    self.cameras[mode].elevation = -25
                elif mode == 'video_block':
                    self.cameras[mode].distance = 2
                    self.cameras[mode].azimuth = 160
                    self.cameras[mode].elevation = -25
                elif mode == 'video_front':
                    self.cameras[mode].distance = 3
                    self.cameras[mode].azimuth = 180
                    self.cameras[mode].elevation = -10

                if self.contact_vis:
                    self.renderers[mode].update_scene(self.data, self.cameras[mode], self.options)
                else:
                    self.renderers[mode].update_scene(self.data, self.cameras[mode])

                pixels = self.renderers[mode].render()
                self.frames[mode].append(pixels)
                self.outs[mode].write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    def take_screenshot(self, time_step, file_name='screenshot'):
        for mode in self.render_modes:
            if mode != 'livecam':
                file_path = os.path.join(self.output_dir, f"{file_name}_{mode}_timestep_{time_step:.4f}.png")
                if self.contact_vis:
                    self.renderers[mode].update_scene(self.data, self.cameras[mode], self.options)
                else:
                    self.renderers[mode].update_scene(self.data, self.cameras[mode])
                
                pixels = self.renderers[mode].render()
                cv2.imwrite(file_path, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                #print(f"Screenshot taken for {mode} at timestep {time_step:.4f} and saved to {file_path}")

    def close(self):
        if 'livecam' in self.render_modes:
            self.viewer.close()
        
        for mode in self.render_modes:
            if mode != 'livecam' and hasattr(self, 'outs') and self.outs[mode].isOpened():
                self.outs[mode].release()
            
            if mode in self.frames:
                self.frames[mode].clear()
            
            if mode in self.renderers:
                del self.renderers[mode]

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


def create_block_bodies(block_positions_orientations, block_mass, block_size,
                        block_solimp, block_solref, block_friction):
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
            <joint name="block_joint" type="free" damping="0"></joint>
            <inertial pos="0 0 0" mass="{block_mass}" diaginertia="0.001 0.001 0.001" />
            <geom name="blue_subbox" size="{block_size[0]} {block_size[1]} {block_size[2]}" pos="0 0 0" type="box" rgba="0.3 0.5 0.8 1" 
            solimp="0.99 0.995 0.0000001 0.5 2" solref="0.005 2" friction="5 0.3 0.001" />
        </body>
        '''

    return block_bodies

def create_ur_model(marker_position=None, 
                    block_positions_orientations=None, 
                    block_mass=0.1, 
                    block_size=[0.04, 0.02, 0.08],
                    block_solimp=[0.99, 0.995, 0.0000001, 0.5, 2],
                    block_solref=[0.005, 2],
                    block_friction=[5, 0.01, 0.001],
                    cone="pyramidal",
                    noslip_iterations=5, 
                    noslip_tolerance=1e-6, 
                    impratio=1, 
                    pad_friction=5,
                    pad_solimp=[0.97, 0.99, 0.001],
                    pad_solref=[0.004, 1]):

    marker_body = ""
    if marker_position is not None:
        marker_body = create_marker_body(marker_position)

    block_bodies = ""
    if block_positions_orientations is not None:
        block_bodies = create_block_bodies(block_positions_orientations, block_mass, block_size,
                                            block_solimp, block_solref, block_friction)

    xml_string = f'''
    <mujoco model="ur10e scene">
    <default>
        <default class="ur10e">
        <material specular="0.5" shininess="0.25"/>
        <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
        <position ctrlrange="-6.2831 6.2831"/>
        <general biastype="affine" ctrlrange="-6.2831 6.2831" gainprm="5000" biasprm="0 -5000 -500"/>
        <default class="size4">
            <joint damping="10"/>
            <general forcerange="-330 330"/>
        </default>
        <default class="size3">
            <joint damping="5"/>
            <general forcerange="-150 150"/>
            <default class="size3_limited">
            <joint range="-3.1415 3.1415"/>
            <general ctrlrange="-3.1415 3.1415"/>
            </default>
        </default>
        <default class="size2">
            <joint damping="2"/>
            <general forcerange="-56 56"/>
        </default>
        <default class="visual">
            <geom type="mesh" contype="0" conaffinity="0" group="2"/>
        </default>
        <default class="collision">
            <geom type="capsule" group="3"/>
            <default class="eef_collision">
            <geom type="cylinder"/>
            </default>
        </default>
        <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
        </default>
        <default class="2f85">
        <mesh scale="0.001 0.001 0.001"/>
        <general biastype="affine"/>

        <joint axis="1 0 0"/>
        <default class="driver">
            <joint range="0 0.8" armature="0.005" damping="0.1" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
        </default>
        <default class="follower">
            <joint range="-0.872664 0.872664" armature="0.001" pos="0 -0.018 0.0065" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
        </default>
        <default class="spring_link">
            <joint range="-0.29670597283 0.8" armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
        </default>
        <default class="coupler">
            <joint range="-1.57 0" armature="0.001" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
        </default>

        <default class="visual_2f85">
            <geom type="mesh" contype="0" conaffinity="0" group="2"/>
        </default>
        <default class="collision_2f85">
            <geom type="mesh" group="3"/>
        <default class="pad_box1">
                <geom mass="0" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375" friction="5"
                    solimp="0.97 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1" condim="6"/>
                </default>
        <default class="pad_box2">
        <geom mass="0" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375" friction="5"
            solimp="0.97 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1" condim="6"/>
            </default>
        </default>
        </default>
    </default>

        <include file="universal_robots_ur10e_2f85/ur10e_2f85.xml"/>

        <option integrator="implicitfast"
                cone="{cone}" 
                noslip_iterations="{noslip_iterations}" 
                noslip_tolerance="{noslip_tolerance}" 
                impratio="{impratio}">
            <flag multiccd="enable"/>
        </option>

        <visual>
            <global offheight="2160" offwidth="3840"/>
            <quality offsamples="8"/>
        </visual>

        <asset>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0" rgba="1 1 1 0.7"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" solimp="0.995 0.999 0.0000001 0.5 2"/>

            {marker_body}
            {block_bodies}
        </worldbody>


        <keyframe>
            <key name="home" qpos="-1.5585076620891312 -2.24059215309729 2.7381982806889287 -2.0608726328910887 -1.570816385660346 0.012994184554737283 0.4977685014807726 -0.0005080874815357948 0.49379563711722124 -0.48450963344132164 0.4980256318035263 0.0008091686346243922 0.4923788223820397 -0.4846690775202845 0.2439992149733207 0.17704907687052862 0.09451305877887498 0.9998676880038513 0.0010406732397370223 -0.016119260465067516 0.0019216990141398692" ctrl="-1.5707963267948966 -2.2364409549094275 2.7310424726607567 -2.0626062324855856 -1.5707963267948966 0.0 225.0" />
        </keyframe>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    </mujoco>'''
    
    return xml_string

        

    