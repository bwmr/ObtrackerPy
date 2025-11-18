# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:27:08 2025

@author:  Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""

import os
from skimage import io
import matplotlib.pyplot as plt



def show_cell_trajectories(unet_path, label_df, min_trajectory_length, experiment_id, save_figure_path):
    
    red_df = label_df[label_df.traj_length>=min_trajectory_length]
    print(red_df.traj_id.unique().shape[0], 'cell trajectories with more than',min_trajectory_length,'frames')
        
    phase_objects = os.listdir(unet_path)
    # masks_paths = [s for s in phase_objects if 'mask' in s]
    phase_paths = [s for s in phase_objects if 'cp_masks' not in s]
    
    # mask_path = [s for s in masks_paths if 't1_mask.tif' in s]
    phase_path = [s for s in phase_paths if 't000.tif' in s]
    
    phase_image = io.imread(unet_path+'/'+phase_path[0])

    plt.figure(figsize=(10,10))
    plt.imshow(phase_image, cmap='gray')    
    for traj in red_df.traj_id.unique():
        traj_df = red_df[red_df.traj_id==traj]
        plt.plot(traj_df['centroid-1'], traj_df['centroid-0'])
    plt.plot([50, 50+5/0.066], [50,50], linewidth=2, color='white')
    plt.text(50,105, '5 μM', fontsize=12, color='white')
    if os.path.isdir(save_figure_path):
        plt.savefig(save_figure_path+'/check_cell_tracking'+experiment_id+'.jpeg')
    plt.show()
    