import time, pdb
import sys
import json
import os

import mujoco
import mujoco.viewer

from datetime import datetime
from env_test_utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'research_main', 'envs'))
sys.path.append(parent_dir)

from mujoco_utils import *
from scene_builder import *

def xml_test():
    xml_string = \
        f'''<mujoco model="ur10e scene">
        <include file="../research_main/envs/universal_robots_ur10e_2f85_example/ur10e_2f85.xml"/>

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
    #try:
    local_time = datetime.now()
    formatted_time = local_time.strftime('%Y-%m-%d_%H-%M')
    #output_dir = create_directories(formatted_time=formatted_time, render_modes="livecam", mode="mujoco_test")

    # current_dir = os.path.dirname(__file__)
    # path_to_model = os.path.join(current_dir, '..', 'research_main', 'envs', 'universal_robots_ur10e_2f85_example', 'ur10e_2f85.xml')
    model = mujoco.MjModel.from_xml_path('ur10e_2f85_mpc.xml')     
    #model.opt.timestep = 0.008   
    
    #model = mujoco.MjModel.from_xml_path("mujoco_menagerie/universal_robots_ur10e/scene.xml")
    data = mujoco.MjData(model)

    mpc_ctrl = []
    mpc_qpos = []
    mpc_qvel = []
    mpc_timestep = 0
    csv_initialized = False
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'research_main/envs/precomputed_mpc_traj', 'interpolated_trajectory.csv')

    # Open the CSV file using the corrected path
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        mpc_ctrl = []
        mpc_qpos = []
        mpc_qvel = []
        for row in reader:
            ctrl_values = [float(x) for x in row['Ctrl'].split(',')]
            qpos_values = [float(x) for x in row['QPos'].split(',')]
            qvel_values = [float(x) for x in row['QVel'].split(',')]
            mpc_ctrl.append(ctrl_values[:6])
            mpc_qpos.append(qpos_values[:6])
            mpc_qvel.append(qvel_values[:6])

    mpc_ctrl = np.array(mpc_ctrl)
    mpc_qpos = np.array(mpc_qpos)
    mpc_qvel = np.array(mpc_qvel)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        num = 0
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:6] = mpc_qpos[0]
        data.qvel[:6] = mpc_qvel[0]
        print("Simulation timestep: ", model.opt.timestep)

        while viewer.is_running() and time.time() - start_time < 5:
            # active_motors.print_actuator_parameters(model)
            step_start = time.time()
            current_time = time.time() - start_time

#========================================================================================
            if mpc_timestep < len(mpc_ctrl):
                given_ctrl = mpc_qpos[mpc_timestep]
                data.ctrl[:6] = given_ctrl

                # Log data at each time step
                if not csv_initialized:
                    with open('mpc_log.csv', 'w', newline='') as csvfile:
                        log_writer = csv.writer(csvfile)
                        log_writer.writerow(['Time', 'Ctrl', 'QPos', 'QVel'])  # Write headers
                    csv_initialized = True  # Prevent rewriting headers

                with open('mpc_log.csv', 'a', newline='') as csvfile:
                    log_writer = csv.writer(csvfile)
                    ctrl_str = ','.join(map(str, data.ctrl[:6].copy()))
                    qpos_str = ','.join(map(str, data.qpos[:6].copy()))
                    qvel_str = ','.join(map(str, data.qvel[:6].copy()))
                    log_writer.writerow([data.time, ctrl_str, qpos_str, qvel_str])

                mpc_timestep += 1
 
#========================================================================================
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    #except:
        
        #plot_joints_data(output_dir, time_datas, qpos_datas, qvel_datas, ctrl_datas)