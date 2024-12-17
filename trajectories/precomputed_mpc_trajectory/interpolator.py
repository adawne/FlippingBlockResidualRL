import numpy as np
from scipy.interpolate import interp1d
import csv

def process_trajectory(interpolate=True, new_timestep=None):
    """
    Process the trajectory by filtering and optionally interpolating the data.
    
    Args:
        interpolate (bool): Whether to interpolate the data. If False, the data is only filtered.
        new_timestep (float): If interpolation is enabled, this defines the new timestep for interpolation.
                              Set to None to disable interpolation.
    """
    # Step 1: Read the original trajectory from 'mpc_trajectory.csv'
    time_log = []
    ctrl_log = []
    qpos_log = []
    qvel_log = []

    with open('mpc_trajectory.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            time_log.append(float(row[0]))
            ctrl_log.append([float(x) for x in row[1].split(',')])  # Ctrl column
            qpos_log.append([float(x) for x in row[2].split(',')])  # QPos column
            qvel_log.append([float(x) for x in row[3].split(',')])  # QVel column

    # Convert lists to numpy arrays
    time_log = np.array(time_log)
    ctrl_log = np.array(ctrl_log)
    qpos_log = np.array(qpos_log)
    qvel_log = np.array(qvel_log)

    # Step 2: Filter the data where time > threshold
    filter_indices = time_log > 6
    time_log_filtered = time_log[filter_indices]
    ctrl_log_filtered = ctrl_log[filter_indices]
    qpos_log_filtered = qpos_log[filter_indices]
    qvel_log_filtered = qvel_log[filter_indices]

    if interpolate and new_timestep is not None:
        # Step 3: Interpolate to the new timestep
        new_time_log = np.arange(time_log_filtered[0], time_log_filtered[-1], new_timestep)

        # Interpolate qpos, qvel, and ctrl to match the new timestep
        ctrl_interp = interp1d(time_log_filtered, ctrl_log_filtered, axis=0, kind='linear')
        qpos_interp = interp1d(time_log_filtered, qpos_log_filtered, axis=0, kind='linear')
        qvel_interp = interp1d(time_log_filtered, qvel_log_filtered, axis=0, kind='linear')

        # Generate the new interpolated trajectory
        new_ctrl_log = ctrl_interp(new_time_log)
        new_qpos_log = qpos_interp(new_time_log)
        new_qvel_log = qvel_interp(new_time_log)

        # Update time, ctrl, qpos, and qvel with the interpolated data
        time_log_final = new_time_log
        ctrl_log_final = new_ctrl_log
        qpos_log_final = new_qpos_log
        qvel_log_final = new_qvel_log
    else:
        # No interpolation, just use the filtered data
        time_log_final = time_log_filtered
        ctrl_log_final = ctrl_log_filtered
        qpos_log_final = qpos_log_filtered
        qvel_log_final = qvel_log_filtered

    # Step 4: Save the processed (filtered and optionally interpolated) trajectory
    output_file = 'interpolated_trajectory.csv' if interpolate else 'filtered_trajectory.csv'
    with open(output_file, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        # Write header
        log_writer.writerow(['Time', 'Ctrl', 'QPos', 'QVel'])
        
        # Write processed rows
        for i in range(len(time_log_final)):
            ctrl_str = ','.join(map(str, ctrl_log_final[i]))
            qpos_str = ','.join(map(str, qpos_log_final[i]))
            qvel_str = ','.join(map(str, qvel_log_final[i]))
            log_writer.writerow([time_log_final[i], ctrl_str, qpos_str, qvel_str])

    print(f"Trajectory saved to '{output_file}'")


process_trajectory(interpolate=False)


