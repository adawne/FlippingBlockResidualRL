import time
import numpy as np
import mujoco
import mujoco.viewer

from utils_push_mujoco import *

def add_sphere_to_model(model_xml, pos, size=1):
    # Add a new body with a sphere geom to the XML string
    sphere_body = f'''
    <body name="sphere" pos="{pos[0]} {pos[1]} {pos[2]}">
        <geom type="sphere" size="{size}" rgba="1 0 0 1"/>
    </body>
    '''
    
    # Insert the new body before the closing </mujoco> tag
    insert_pos = model_xml.find('</mujoco>')
    model_xml = model_xml[:insert_pos] + sphere_body + model_xml[insert_pos:]
    
    return model_xml

if (__name__=='__main__'):

    target_position = [-0.587, -0.094, 1.25]
    target_orientation = [0, 0, 0]
    
    world_xml_model = create_ur_model(marker_position=None)
    
    # Add a sphere to the model
    sphere_pos = [0, 1, 0]
    modified_model_xml = add_sphere_to_model(world_xml_model, sphere_pos)
    
    model = mujoco.MjModel.from_xml_string(modified_model_xml)
    data = mujoco.MjData(model)
    key_id = model.key("home").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = 0.1
        start = time.time()

        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()
            target_q = diffik(model, data, target_position, target_orientation)
            close_gripper(data)
            print(data.ctrl[6])     
            mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
