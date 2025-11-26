"""
Created on Mon Nov 17 22:27:08 2025.

@author:  Alexandros Papagiannakis, HHMI @Stanford University, 2025

"""


import matplotlib.pyplot as plt


def show_cell_trajectories(label_df, min_trajectory_length):
    """Plot cell trajectories with given minimal length."""
    red_df = label_df[label_df.traj_length>=min_trajectory_length]
    print(red_df.traj_id.unique().shape[0],
          'cell trajectories with more than',min_trajectory_length,'frames')

    plt.figure(figsize=(3,3))
    for traj in red_df.traj_id.unique():
        traj_df = red_df[red_df.traj_id==traj]
        plt.plot(traj_df.frame, traj_df.area, color='gray',linewidth = 0.5)
    plt.xlabel('Time (frame)')
    plt.ylabel('Object area (px)')
    plt.yscale('log')
    plt.show()
