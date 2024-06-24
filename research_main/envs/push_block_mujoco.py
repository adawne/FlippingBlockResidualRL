import time
import numpy as np
import mujoco
import mujoco.viewer

from utils_push_mujoco import *
from finite_state_machine import *

if (__name__=='__main__'):

    block_positions_orientations = [
        ([0.5, 0.5, 0.2], [np.pi/2, 0, 0]),
    ]


    world_xml_model = create_ur_model(marker_position=None, block_positions_orientations=block_positions_orientations)
    
    model = mujoco.MjModel.from_xml_string(world_xml_model)
    data = mujoco.MjData(model)
    #key_id = model.key("home").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = 0.1
        start = time.time()

        fsm = FiniteStateMachine()

        actuator_ids = [0, 1, 2, 3, 4, 5]
        joint_motors = ActuatorController(actuator_ids)
        joint_motors.set_position_controller()
        joint_motors.update_actuator(model)

        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            current_position, _ = get_ee_pose(model, data)
            target_position, target_orientation, move_to_next_state, iteration = fsm.update(data, current_position)

            joint_angles_target = diffik(model, data, target_position, target_orientation)
            data.ctrl[:6] = joint_angles_target[:6]
            mujoco.mj_step(model, data)

            #print(np.linalg.norm(np.subtract(joint_angles_target[:6], get_joint_angles(data)[:6])))
            #print(get_block_pose(model, data, 'block0'))

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                
            ## Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            ## Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
