import time, pdb

import json

import mujoco
import mujoco.viewer

def create_world_xml_model(object_xml_models):
    xml_string = \
        f'''<mujoco>
        <option gravity="0 0 -10"/>
        <compiler angle="radian"/>
        <asset>
        <texture name="plane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
        width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true"/>
        </asset>
        
        <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="plane" condim="3"/>'''

    for object_xml_model in object_xml_models:
        xml_string += object_xml_model
        
    xml_string += \
        f'''</worldbody>
        </mujoco>'''

    return xml_string

def create_cube_model(cube_name, center, coordinates):

    xml_string = \
        f'''
        <body name="{cube_name}" pos="{center[0]} {center[1]} {center[2]}">
        <freejoint name="{cube_name}_joint"/>'''

    for coordinate in coordinates:
        position = (.1*coordinate[0]+.05, .1*coordinate[1]+.05, .1*coordinate[2]+.05)
        xml_string += \
            f'''
            <geom size="0.05 0.05 0.05" pos="{position[0]} {position[1]} {position[2]}" type="box"/>
            '''
        # <geom size="0.05 0.05 0.05" pos="0.15 0.05 0.05" type="box"/>
        # <geom size="0.05 0.05 0.05" pos="0.15 -0.05 0.05" type="box"/>

    xml_string += \
        f'''
        </body>
        '''

    return xml_string

def xml_test():
    xml_string = \
        f'''<mujoco model="ur10e scene">
        <include file="universal_robots_ur10e_2f85_example/ur10e_2f85.xml"/>

        <statistic center="0.4 0 0.4" extent="1"/>
        
        <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
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
        </worldbody>
        </mujoco>'''
    
    return xml_string


if (__name__=='__main__'):

    # with open('./cube_model/coordinates.json') as json_file:
    #     multiple_cubes_coordinates = json.load(json_file)

    # # print(cube_coordinates)

    # cube_xml_models = []
    # for i, cube_coordinates in enumerate(multiple_cubes_coordinates):
    #     cube_xml_model = create_cube_model(cube_name='cube_{}'.format(i),
    #                                        center=[0, 0, 1],
    #                                        coordinates = cube_coordinates)

    #     cube_xml_models.append(cube_xml_model)

    # world_xml_model = create_world_xml_model(cube_xml_models)

    world_xml_model = xml_test()

    # print(world_xml_model)
    
    model = mujoco.MjModel.from_xml_string(world_xml_model)

    data = mujoco.MjData(model)
    #print("Name of geom: ", model.geom(45).name, model.geom(57).name)
    # print(data.geom_xpos)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        num = 0

        # # Function to make the window full screen
        # def set_fullscreen(window):
        #     monitor = glfw.get_primary_monitor()
        #     mode = glfw.get_video_mode(monitor)
        #     glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)

        # # Set the viewer window to full screen
        # set_fullscreen(viewer.window)

        while viewer.is_running() and time.time() - start < 300:
            
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            # data.xpos[0][0] += .1
            # my_force = [0,0,.0]
            # my_torque = [0,0,0]
            # point_on_body = [0,0,0]
            # bodyid = 1
            
            # mujoco.mj_applyFT(model, data, my_force, my_torque, point_on_body, bodyid, data.qfrc_applied)

            # data.xfrc_applied[1][2] = 30
            # data.xfrc_applied[1][4] = .001

            # print(data.xfrc_applied)
            
            mujoco.mj_step(model, data)
            # print(data.geom_xpos)
            # num += 1
            # print(num)

            # pdb.set_trace()


            # Example modification of a viewer option: toggle contact points every two seconds.
            #with viewer.lock():
            #    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
