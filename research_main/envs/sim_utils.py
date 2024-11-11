import os
import numpy as np
import json
import csv
import itertools
import matplotlib.pyplot as plt
import re
import pandas as pd

from scipy.spatial.transform import Rotation as R

def create_directories(i, block_mass, formatted_time, render_modes):
    if "livecam" not in render_modes:
        output_dir = f'outputs/{formatted_time}_{render_modes}'
        sub_output_dir = f'outputs/{formatted_time}_{render_modes}/{i}_{block_mass:.3f}'
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(sub_output_dir, exist_ok=True)
        
        return output_dir, sub_output_dir

    else:
        return None, None

def create_directories_and_save_config(i, block_mass, formatted_time, render_modes, config):
    if "livecam" not in render_modes:
        output_dir = f'outputs/{formatted_time}_{render_modes}'
        sub_output_dir = f'outputs/{formatted_time}_{render_modes}/{i}_{block_mass:.3f}'
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(sub_output_dir, exist_ok=True)
        
        config_filename = os.path.join(sub_output_dir, 'config.json')
        save_config(config, config_filename)
        
        return output_dir, sub_output_dir
    else:
        return None, None

def build_config(i, render_modes, contact_vis, random_mass, block_mass, block_size, ee_flip_target_velocity, 
                 sim_description, block_solimp, block_solref, block_friction, cone, noslip_iterations, 
                 noslip_tolerance, impratio, pad_friction, pad_solimp, pad_solref, clampness, use_random_parameters):
    config = {
        'iteration': i,
        'render_modes': render_modes,
        'contact_vis': contact_vis,
        'random_mass': random_mass,
        'block_mass': block_mass,
        'block_size': block_size,
        'ee_flip_target_velocity': ee_flip_target_velocity,
        'sim_description': sim_description,
        'block_solimp': block_solimp,
        'block_solref': block_solref,
        'block_friction': block_friction,
        'cone': cone,
        'noslip_iterations': noslip_iterations,
        'noslip_tolerance': noslip_tolerance,
        'impratio': impratio,
        'pad_friction': pad_friction,
        'pad_solimp': pad_solimp,
        'pad_solref': pad_solref,
        'clampness': clampness,
        'use_random_parameters': use_random_parameters
    }

    return config

def analyze_contacts(data, model, fsm, time):
    contact_info = [time]  
    for j in range(data.ncon):
        geom1_name = model.geom(data.contact[j].geom1).name
        geom2_name = model.geom(data.contact[j].geom2).name
        distance = data.contact[j].dist
        pos = data.contact[j].pos
        contact_info.append((geom1_name, geom2_name, distance, pos))
    return contact_info

def perform_discrepancy_analysis(release_time, touch_ground_time, block_release_pos, block_release_transvel, 
                                 block_touch_ground_height, block_release_quat, block_touch_ground_quat, block_touch_ground_velocity, 
                                 time_hist, block_position_hist, block_ang_vel_hist):
    return check_physical_assumptions(
        release_time=release_time, 
        touch_ground_time=touch_ground_time,
        block_release_pos=block_release_pos, 
        block_release_transvel=block_release_transvel, 
        block_release_quat=block_release_quat, 
        block_touch_ground_height= block_touch_ground_height, 
        block_touch_ground_quat=block_touch_ground_quat, 
        block_touch_ground_velocity=block_touch_ground_velocity,
        time_hist=time_hist,
        block_position_hist=block_position_hist,
        block_ang_vel_hist=block_ang_vel_hist
    )


def plot_and_save_results(sub_output_dir, iteration, release_time, time_hist, fsm, block_trans_vel_hist, landing_time_pred, 
                          touch_ground_time, steady_time, block_position_hist, block_orientation_hist, 
                          block_ang_vel_hist, block_release_pos, block_release_orientation, block_release_transvel, 
                          block_release_angvel, block_touch_ground_position, block_touch_ground_orientation, 
                          block_steady_position, block_steady_orientation):
    plot_velocity_ee(sub_output_dir, release_time, time_hist, fsm.ee_vel_hist)
    plot_velocity_comparison(sub_output_dir, release_time, time_hist, fsm.ee_vel_hist, block_trans_vel_hist)
    plot_joint_velocities(sub_output_dir, release_time, fsm.time_hist, fsm.joint_vel_hist, fsm.target_joint_vel_hist)
    plot_block_pose(sub_output_dir, release_time, landing_time_pred, touch_ground_time, steady_time, time_hist, 
                    block_position_hist, block_orientation_hist, block_trans_vel_hist, block_ang_vel_hist)
    
    save_iter_stats(sub_output_dir, iteration, release_time, touch_ground_time, steady_time, fsm.release_ee_velocity, 
                    block_release_pos, block_release_orientation, block_release_transvel, block_release_angvel, 
                    block_touch_ground_position, block_touch_ground_orientation, block_steady_position, 
                    block_steady_orientation, block_position_hist)

def plot_and_save_contacts(sub_output_dir, contact_hist):
    contacts_csv_path = os.path.join(sub_output_dir, 'contacts.csv')  # This ensures a correct file path
    print(f"Saving contacts to {contacts_csv_path}")
    
    # Ensure the directory exists
    os.makedirs(sub_output_dir, exist_ok=True)
    
    # Check if contact_hist is empty
    if not contact_hist:
        print("Warning: contact_hist is empty, no contacts to save.")
        return

    # Save contacts to CSV
    try:
        save_contacts_to_csv(sub_output_dir, contact_hist)  # Pass only the directory here
        print(f"Contacts saved successfully to {contacts_csv_path}")
        
        # Load and plot contacts
        contacts_csv = pd.read_csv(contacts_csv_path)
        plot_contacts_data(sub_output_dir, contacts_csv)
        
    except Exception as e:
        print(f"Error while saving contacts: {e}")



def log_simulation_results(i, release_time, fsm, block_release_pos, block_release_orientation, 
                           block_release_transvel, block_release_angvel, touch_ground_time, 
                           block_touch_ground_position, block_touch_ground_orientation):
    print("="*91)
    print(f"Iteration: {i}")
    print(f"Block release time: {release_time}")
    print(f"Release EE velocity: {fsm.release_ee_velocity}")
    print(f"Block release position: {block_release_pos}")
    print(f"Block release orientation: {block_release_orientation}")
    print(f"Block translational release velocity: {block_release_transvel}")
    print(f"Block angular release velocity: {block_release_angvel}")
    print("-"*91)
    print(f"Block touch the ground time: {touch_ground_time}")
    print(f"Position when the block touched the ground: {block_touch_ground_position}")
    print(f"Orientation when the block touched the ground: {block_touch_ground_orientation}")


def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def save_contacts_to_csv(output_dir, contact_hist):
    # Assuming output_dir is the directory path, not a file path
    filename = os.path.join(output_dir, 'contacts.csv')  
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Contacts"]) 
        for record in contact_hist:
            time = record[0]
            contacts = record[1:]
            contact_info_str = "; ".join([f"{geom1} - {geom2} (Distance: {dist:.4f} - Pos: {pos})" 
                                          for geom1, geom2, dist, pos in contacts])
            writer.writerow([time, contact_info_str])

def preprocess_contacts_data(df):
    # Create an empty list to store the parsed data
    parsed_data = []

    # Regex pattern to extract contact information
    pattern = re.compile(r'(?P<geom1>\w+) - (?P<geom2>\w+) \(Distance: (?P<distance>[-+]?[0-9]*\.?[0-9]+) - Pos: \[(?P<pos>[^\]]+)\]\)')

    # Loop over each row and parse the contacts
    for _, row in df.iterrows():
        time = row['Time']
        contacts_str = row['Contacts']
        
        # Find all matches for contacts
        matches = pattern.findall(contacts_str)
        
        for match in matches:
            geom1, geom2, distance, pos = match
            distance = float(distance)
            pos = [float(p) for p in pos.split()]
            parsed_data.append([time, geom1, geom2, distance, pos])

    # Create a DataFrame from the parsed data
    parsed_df = pd.DataFrame(parsed_data, columns=['Time', 'Geom1', 'Geom2', 'Distance', 'Pos'])

    # Split the position column into three separate columns for x, y, z
    parsed_df[['Pos_X', 'Pos_Y', 'Pos_Z']] = pd.DataFrame(parsed_df['Pos'].tolist(), index=parsed_df.index)

    # Drop the original Pos column as it's no longer needed
    parsed_df.drop(columns=['Pos'], inplace=True)

    return parsed_df

def plot_contacts_data(output_dir, df):
    df = preprocess_contacts_data(df)
    
    df['Contact_Order'] = df.groupby(['Time', 'Geom1', 'Geom2']).cumcount() + 1
    df['Contact_Label'] = df['Geom1'] + ' - ' + df['Geom2'] + ' (' + df['Contact_Order'].astype(str) + ')'
    
    unique_contacts = df['Geom1'] + ' - ' + df['Geom2']
    
    for contact_pair in unique_contacts.unique():
        contact_df = df[unique_contacts == contact_pair]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        for contact_label in contact_df['Contact_Label'].unique():
            contact_data = contact_df[contact_df['Contact_Label'] == contact_label]
            
            # Plot X position
            axs[0].plot(contact_data['Time'], contact_data['Pos_X'], label=f'{contact_label}')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Position X')
            axs[0].legend()

            # Plot Y position
            axs[1].plot(contact_data['Time'], contact_data['Pos_Y'], label=f'{contact_label}')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Position Y')
            axs[1].legend()

            # Plot Z position
            axs[2].plot(contact_data['Time'], contact_data['Pos_Z'], label=f'{contact_label}')
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Position Z')
            axs[2].legend()

        contact_pair_cleaned = contact_pair.replace(' ', '_').replace('/', '_')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/contact_position_plot_{contact_pair_cleaned}.png')
        plt.close()

def save_sim_stats(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, 
                   landing_velocities_discrepancies, block_release_ver_velocities):
    filename = f'{output_dir}/results.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Mass', 'Time Discrepancy (%)', 'Angle Discrepancy X (%)', 'Angle Discrepancy Y (%)', 'Angle Discrepancy Z (%)', 
            'Height Discrepancy (%)','Landing Velocity Discrepancy (%)', 'Block Release Ver Velocity (m/s)'
        ])  
        for mass, time_discrepancy, angle_discrepancy, height_discrepancy, landing_velocity_discrepancy, ver_velocity in zip(masses, 
        time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocities):
            writer.writerow([
                mass, time_discrepancy, angle_discrepancy[0], angle_discrepancy[1], angle_discrepancy[2],  
                height_discrepancy, landing_velocity_discrepancy, ver_velocity
            ])


def save_iter_stats(output_dir, iteration, release_time, block_touch_ground_time, block_steady_state_time, ee_velocity, block_release_pos, 
                    block_release_orientation, block_release_transvel, block_release_angvel, block_touch_ground_position, 
                    block_touch_ground_orientation, block_steady_position, block_steady_orientation, 
                    block_position_hist):
    
    time_in_air = block_touch_ground_time - release_time
    
    block_heights = [pos[2] for pos in block_position_hist]  
    lowest_height = min(block_heights)

    filename = f'{output_dir}/results.txt'
    with open(filename, 'w') as file:
        file.write(f"Iteration: {iteration}\n")
        file.write(f"Block release time: {release_time}\n")
        file.write(f"Block touch the ground time: {block_touch_ground_time}\n")
        file.write(f"Block steady state time: {block_steady_state_time}\n")
        file.write(f"Time in air (touch ground - release): {time_in_air}\n")
        file.write(f"Release EE velocity: {ee_velocity}\n")
        file.write(f"Block release position: {block_release_pos}\n")
        file.write(f"Block release orientation: {block_release_orientation}\n")
        file.write(f"Block translational release velocity: {block_release_transvel}\n")
        file.write(f"Block angular release velocity: {block_release_angvel}\n")
        file.write(f"Position when the block touched the ground: {block_touch_ground_position}\n")
        file.write(f"Orientation when the block touched the ground: {block_touch_ground_orientation}\n")
        file.write(f"Position when the block landed steadily: {block_steady_position}\n")
        file.write(f"Orientation when the block landed steadily: {block_steady_orientation}\n")
        file.write(f"Lowest block height: {lowest_height:.4f} m\n")  # Print the lowest height



def check_physical_assumptions(release_time, touch_ground_time, block_release_pos, block_release_transvel, block_release_quat, 
                                block_touch_ground_height, block_touch_ground_quat, block_touch_ground_velocity, time_hist, 
                                block_position_hist, block_ang_vel_hist, g=9.81):
    
    block_release_ver_velocity = block_release_transvel[2]

    block_heights = [pos[2] for pos in block_position_hist] 
    highest_height_exp = max(block_heights)
    highest_height_theory = block_release_pos[2] + (block_release_ver_velocity**2 / (2 * g))

    height_discrepancy = np.abs(highest_height_exp - highest_height_theory)
    height_discrepancy_percentage = (height_discrepancy / highest_height_theory) * 100

    time_in_air_exp = touch_ground_time - release_time
    time_ascent = (block_release_ver_velocity / g) 
    time_descent_first = block_release_ver_velocity**2/g**2
    time_descent_second = 2*(block_release_pos[2]-block_touch_ground_height)/g

    time_descent = np.sqrt(time_descent_first + time_descent_second)
    time_in_air_theory =  time_ascent + time_descent
    time_discrepancy = np.abs(time_in_air_exp - time_in_air_theory)
    time_discrepancy_percentage = (time_discrepancy / time_in_air_theory) * 100
    

    delta_y = block_release_pos[2] - block_touch_ground_height 
    landing_velocity_theory = np.sqrt(block_release_transvel[2]**2 + 2 * g * delta_y)


    #landing_velocity_theory = np.sqrt(-(block_release_transvel[0]**2 - 2 * g * (block_release_pos[2] - block_touch_ground_height)))
    landing_velocity_exp = np.linalg.norm(block_touch_ground_velocity)
    landing_velocity_discrepancy = np.abs(landing_velocity_exp - landing_velocity_theory)
    landing_velocity_discrepancy_percentage = (landing_velocity_discrepancy / landing_velocity_theory) * 100

    # Find in-air indices
    in_air_indices = [i for i, t in enumerate(time_hist) if release_time <= t <= touch_ground_time]

    if in_air_indices:
        omega_exp = np.mean(np.array(block_ang_vel_hist)[in_air_indices], axis=0)
    else:
        omega_exp = np.zeros_like(block_release_orientation)
    
    release_quat = np.asarray(block_release_quat).flatten()
    touch_ground_quat = np.asarray(block_touch_ground_quat).flatten()

    # Convert to Rotation objects
    release_rot = R.from_quat(release_quat)
    touch_ground_rot = R.from_quat(touch_ground_quat)

    delta_rotation = R.from_rotvec(omega_exp * time_in_air_exp)  # Rotation vector -> Rotation matrix
    final_rotation_theory_quat = release_rot * delta_rotation


    final_rotation_theory_euler = final_rotation_theory_quat.as_euler('xyz', degrees=True)

    final_orientation_exp_euler = R.from_quat(touch_ground_quat).as_euler('xyz', degrees=True)

    theta_final_discrepancy_degrees = final_rotation_theory_euler - final_orientation_exp_euler
    theta_final_discrepancy_degrees = (theta_final_discrepancy_degrees + 180) % 360 - 180
    theta_final_discrepancy_percentage = (np.abs(theta_final_discrepancy_degrees) / 360.0) * 100.0


    print("-" * 91)
    print("Testing Physical Assumptions")
    print("-" * 91)
    print(f"Time in the air (experimental): {time_in_air_exp:.4f} s")
    print(f"Time in the air (theoretical): {time_in_air_theory:.4f} s")
    print(f"Discrepancy in time: {time_discrepancy:.4f} s ({time_discrepancy_percentage:.2f}%)")
    print("-" * 91)
    print(f"Highest block altitude (experimental): {highest_height_exp:.4f} m")
    print(f"Highest block altitude (theoretical): {highest_height_theory:.4f} m")
    print(f"Discrepancy in altitude: {height_discrepancy:.4f} m ({height_discrepancy_percentage:.2f}%)")
    print("-" * 91)
    print(f"Landing velocity (experimental): {landing_velocity_exp:.4f} m/s")
    print(f"Landing velocity (theoretical): {landing_velocity_theory:.4f} m/s")
    print(f"Discrepancy in landing velocity: {landing_velocity_discrepancy:.4f} m/s ({landing_velocity_discrepancy_percentage:.2f}%)")
    print("-" * 91)
    print(f"Average angular velocity: {omega_exp}")
    print(f"Final orientation (experimental, Euler): {final_orientation_exp_euler}")
    print(f"Final orientation (theoretical, Euler): {final_rotation_theory_euler}")
    print(f"Discrepancy in final orientation (degrees): {theta_final_discrepancy_degrees}")
    print(f"Discrepancy in final orientation (percentage): {theta_final_discrepancy_percentage}%")
    print("=" * 91)

    return time_discrepancy_percentage, theta_final_discrepancy_percentage, height_discrepancy_percentage, landing_velocity_discrepancy_percentage
        

def plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocity):
    plt.figure(figsize=(8, 30)) 

    # Plot Time Discrepancy
    plt.subplot(5, 1, 1)
    plt.plot(masses, time_discrepancies, 'o-', color='blue', label='Time Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Time Discrepancy (%)')
    plt.title('Mass vs Time Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Angle Discrepancy
    plt.subplot(5, 1, 2)
    angle_discrepancies = np.array(angle_discrepancies)
    plt.plot(masses, angle_discrepancies[:, 0], 'o-', color='red', label='Angle Discrepancy X')
    plt.plot(masses, angle_discrepancies[:, 1], 'o-', color='green', label='Angle Discrepancy Y')
    plt.plot(masses, angle_discrepancies[:, 2], 'o-', color='orange', label='Angle Discrepancy Z')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Angle Discrepancy (%)')
    plt.title('Mass vs Angle Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Block Release Vertical Velocity
    plt.subplot(5, 1, 3)
    plt.plot(masses, block_release_ver_velocity, 'o-', color='purple', label='Block Release Vertical Velocity')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.title('Mass vs Block Release Vertical Velocity')
    plt.grid(True)
    plt.legend()

    # Plot Height Discrepancy
    plt.subplot(5, 1, 4)
    plt.plot(masses, height_discrepancies, 'o-', color='brown', label='Height Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Height Discrepancy (%)')
    plt.title('Mass vs Height Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Landing Velocity Discrepancy
    plt.subplot(5, 1, 5)
    plt.plot(masses, landing_velocities_discrepancies, 'o-', color='cyan', label='Landing Velocity Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Landing Velocity Discrepancy (%)')
    plt.title('Mass vs Landing Velocity Discrepancy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'discrepancy_vs_mass.png')
    plt.savefig(filename)
    plt.close()

def save_qpos_qvel(qpos, qvel, iteration=None):
    current_dir = os.getcwd()
    
    qpos_filename = os.path.join(current_dir, f'qpos_{iteration if iteration is not None else "latest"}.txt')
    qvel_filename = os.path.join(current_dir, f'qvel_{iteration if iteration is not None else "latest"}.txt')
    
    np.savetxt(qpos_filename, qpos, delimiter=',', header='qpos', comments='')
    
    np.savetxt(qvel_filename, qvel, delimiter=',', header='qvel', comments='')


def generate_simulation_parameters():
    block_mass = [0.1]  
    print(f"Number of block_mass combinations: {len(block_mass)}")

    # block_friction: take the lowest and highest values
    block_friction_first = [1, 10]
    block_friction_second = [round(0.005, 3), round(0.05, 3)]
    block_friction_third = [1e-4, 0.1]
    block_friction = list(itertools.product(block_friction_first, block_friction_second, block_friction_third))
    print(f"Number of block_friction combinations: {len(block_friction)}")

    cone = ['pyramidal', 'elliptic']
    print(f"Number of cone combinations: {len(cone)}")

    noslip_iterations = [1, 5]
    print(f"Number of noslip_iterations combinations: {len(noslip_iterations)}")

    noslip_tolerance = [1e-6, 1e-10]
    print(f"Number of noslip_tolerance combinations: {len(noslip_tolerance)}")

    impratio = [1, 5]
    print(f"Number of impratio combinations: {len(impratio)}")

    pad_friction = [1, 5]
    print(f"Number of pad_friction combinations: {len(pad_friction)}")

    pad_solimp_first = [0.97, 0.99]  
    pad_solimp_second = [0.99, 0.999]
    pad_solimp_third = 0.001

    pad_solimp = [(first, second, pad_solimp_third) 
                  for first, second in itertools.product(pad_solimp_first, pad_solimp_second)]
    print(f"Number of pad_solimp combinations: {len(pad_solimp)}")

    pad_solref_first = [round(0.004, 3), round(0.02, 3)]
    pad_solref_second = [1, 2]
    pad_solref = list(itertools.product(pad_solref_first, pad_solref_second))
    print(f"Number of pad_solref combinations: {len(pad_solref)}")

    clampness = [250]
    print(f"Number of clampness combinations: {len(clampness)}")

    # Combine all parameters excluding solimp and solref using itertools.product
    all_combinations = itertools.product(
        block_mass, block_friction, cone, noslip_iterations, noslip_tolerance,
        impratio, pad_friction, pad_solimp, pad_solref, clampness
    )

    num_combinations = (
        len(block_mass) *
        len(block_friction) *
        len(cone) *
        len(noslip_iterations) *
        len(noslip_tolerance) *
        len(impratio) *
        len(pad_friction) *
        len(pad_solimp) *
        len(pad_solref) *
        len(clampness)
    )

    print(f"Total number of combinations: {num_combinations}")

    return all_combinations


def plot_block_pose(output_dir, release_time, landing_time_pred, touch_ground_time, steady_time, time_hist, block_position_hist, block_orientation_hist, block_trans_vel_hist, block_ang_vel_hist):
    # Process the data for plotting
    block_position_hist = list(zip(*block_position_hist))  # Transpose to separate x, y, z components
    block_orientation_hist = list(zip(*block_orientation_hist))  # Transpose to separate x, y, z components
    block_trans_vel_hist = list(zip(*block_trans_vel_hist))  # Transpose to separate x, y, z components
    block_ang_vel_hist = list(zip(*block_ang_vel_hist))  # Transpose to separate x, y, z components

    plt.figure(figsize=(16, 12))

    # Plotting block position history
    plt.subplot(4, 1, 1)
    plt.plot(time_hist, block_position_hist[0], label='X position')
    plt.plot(time_hist, block_position_hist[1], label='Y position')
    plt.plot(time_hist, block_position_hist[2], label='Z position')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Block Position History')
    plt.legend()

    # Plotting block orientation history
    plt.subplot(4, 1, 2)
    plt.plot(time_hist, block_orientation_hist[0], label='X orientation')
    plt.plot(time_hist, block_orientation_hist[1], label='Y orientation')
    plt.plot(time_hist, block_orientation_hist[2], label='Z orientation')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation')
    plt.title('Block Orientation History')
    plt.legend()

    # Plotting translational velocity qvel[14:17]
    plt.subplot(4, 1, 3)
    plt.plot(time_hist, block_trans_vel_hist[0], label='Vx')
    plt.plot(time_hist, block_trans_vel_hist[1], label='Vy')
    plt.plot(time_hist, block_trans_vel_hist[2], label='Vz')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Translational Velocity qvel[14:17]')
    plt.legend()

    # Plotting angular velocity qvel[17:20]
    plt.subplot(4, 1, 4)
    plt.plot(time_hist, block_ang_vel_hist[0], label='Omega x')
    plt.plot(time_hist, block_ang_vel_hist[1], label='Omega y')
    plt.plot(time_hist, block_ang_vel_hist[2], label='Omega z')
    # Add vertical lines
    plt.axvline(x=release_time, color='g', linestyle='--', label='Release time')
    plt.axvline(x=landing_time_pred, color='r', linestyle='--', label='Predicted landing time')
    plt.axvline(x=touch_ground_time, color='b', linestyle='--', label='Touch ground time')
    plt.axvline(x=steady_time, color='orange', linestyle='--', label='Steady state time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity qvel[17:20]')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'block_flying_poses.png'))

def plot_joint_velocities(output_dir, release_time, time_hist, joint_vel_hist, target_joint_vel_hist):
    joint_vel_hist_np = np.array(joint_vel_hist)
    target_joint_vel_hist_np = np.array(target_joint_vel_hist)
    time_hist_np = np.array(time_hist)

    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

    fig, axs = plt.subplots(6, 1, figsize=(10, 15))  # 6 rows, 1 column of subplots

    for i in range(joint_vel_hist_np.shape[1]):
        axs[i].plot(time_hist_np, joint_vel_hist_np[:, i], label=f'Real {joint_names[i]} Velocity')
        axs[i].plot(time_hist_np, target_joint_vel_hist_np[:, i], linestyle='--', label=f'Target {joint_names[i]} Velocity')
        
        # Plot the vertical line at release time
        axs[i].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Velocity (rad/s)')
        axs[i].set_title(f'{joint_names[i]} Velocity Over Time')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_velocities_subplot.png'))

def plot_velocity_ee(output_dir, release_time, time_hist, end_effector_vel_hist):
    time_hist_np = np.array(time_hist[1:])
    end_effector_vel_hist_np = np.array(end_effector_vel_hist[1:])  # Shape: (num_samples-1, 3)

    velocity_components = ['X', 'Y', 'Z']

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column of subplots (one for each axis)

    for i in range(3):
        axs[i].plot(time_hist_np, end_effector_vel_hist_np[:, i], label=f'End Effector {velocity_components[i]} Velocity')
        
        axs[i].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Velocity (m/s)')
        axs[i].set_title(f'{velocity_components[i]} Velocity Over Time')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'end_effector_velocity.png'))

def plot_velocity_comparison(output_dir, release_time, time_hist, end_effector_vel_hist, block_vel_hist):
    # Ignore the first element of each list
    time_hist_np = np.array(time_hist[1:])
    end_effector_vel_hist_np = np.array(end_effector_vel_hist[1:])  # Shape: (num_samples-1, 3)
    block_vel_hist_np = np.array(block_vel_hist[1:])  # Shape: (num_samples-1, 3)

    velocity_components = ['X', 'Y', 'Z']

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))  # 3 rows, 2 columns of subplots (position and discrepancy for each axis)

    for i in range(3):
        axs[i, 0].plot(time_hist_np, end_effector_vel_hist_np[:, i], label=f'End Effector {velocity_components[i]} Velocity')
        axs[i, 0].plot(time_hist_np, block_vel_hist_np[:, i], linestyle='--', label=f'Block {velocity_components[i]} Velocity')
        
        axs[i, 0].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Velocity (m/s)')
        axs[i, 0].set_title(f'{velocity_components[i]} Velocity Over Time')
        axs[i, 0].legend()

        velocity_discrepancy = end_effector_vel_hist_np[:, i] - block_vel_hist_np[:, i]
        axs[i, 1].plot(time_hist_np, velocity_discrepancy, color='purple', label=f'{velocity_components[i]} Velocity Discrepancy')
        
        axs[i, 1].axvline(x=release_time, color='r', linestyle=':', label='Release Time')

        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Discrepancy (m/s)')
        axs[i, 1].set_title(f'{velocity_components[i]} Velocity Discrepancy (End Effector - Block)')
        axs[i, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'end_effector_block_velocity_comparison.png'))

def plot_discrepancy_vs_mass(output_dir, masses, time_discrepancies, angle_discrepancies, height_discrepancies, landing_velocities_discrepancies, block_release_ver_velocity):
    plt.figure(figsize=(8, 30))  # Increased the figure size to accommodate the extra subplot

    # Plot Time Discrepancy
    plt.subplot(5, 1, 1)
    plt.plot(masses, time_discrepancies, 'o-', color='blue', label='Time Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Time Discrepancy (%)')
    plt.title('Mass vs Time Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Angle Discrepancy
    plt.subplot(5, 1, 2)
    angle_discrepancies = np.array(angle_discrepancies)
    plt.plot(masses, angle_discrepancies[:, 0], 'o-', color='red', label='Angle Discrepancy X')
    plt.plot(masses, angle_discrepancies[:, 1], 'o-', color='green', label='Angle Discrepancy Y')
    plt.plot(masses, angle_discrepancies[:, 2], 'o-', color='orange', label='Angle Discrepancy Z')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Angle Discrepancy (%)')
    plt.title('Mass vs Angle Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Block Release Vertical Velocity
    plt.subplot(5, 1, 3)
    plt.plot(masses, block_release_ver_velocity, 'o-', color='purple', label='Block Release Vertical Velocity')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.title('Mass vs Block Release Vertical Velocity')
    plt.grid(True)
    plt.legend()

    # Plot Height Discrepancy
    plt.subplot(5, 1, 4)
    plt.plot(masses, height_discrepancies, 'o-', color='brown', label='Height Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Height Discrepancy (%)')
    plt.title('Mass vs Height Discrepancy')
    plt.grid(True)
    plt.legend()

    # Plot Landing Velocity Discrepancy
    plt.subplot(5, 1, 5)
    plt.plot(masses, landing_velocities_discrepancies, 'o-', color='cyan', label='Landing Velocity Discrepancy')
    plt.xlabel('Block Mass (kg)')
    plt.ylabel('Landing Velocity Discrepancy (%)')
    plt.title('Mass vs Landing Velocity Discrepancy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'discrepancy_vs_mass.png')
    plt.savefig(filename)
    plt.close()



