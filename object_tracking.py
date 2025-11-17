# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:34:14 2024

@author: alexpapa
"""

import os
from skimage import io
from skimage.measure import regionprops_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def load_tif_images(unet_path, time_string_length=4, image_suffix_str='_cp_masks'):
    
    phase_objects = os.listdir(unet_path)
    masks_paths = [s for s in phase_objects if 'cp_masks' in s]
    
    masked_images = {}
    
    for msk in masks_paths:
        tm = msk[msk.find(image_suffix_str)-time_string_length:msk.find(image_suffix_str)]
        print('reading image timepoint:', tm)
        masked_images[int(tm)] = io.imread(unet_path+'/'+msk)
    
    return masked_images



def apply_drift_correction(masks_dict, cum_drift_x, cum_drift_y):
    
    # The drift is computed in the https://github.com/alexSysBio/UnDrift/blob/main/remove_image_drift.py repository
    
    aligned = {}
    
    keys = sorted(masks_dict.keys())
    shapes = [masks_dict[k].shape for k in keys]
    H, W = shapes[0][0], shapes[0][1]
    
    iy = cum_drift_y
    ix = cum_drift_x
    
    y0 = int(np.max(iy))
    y1 = int(H + np.min(iy))
    x0 = int(np.max(ix))
    x1 = int(W + np.min(ix))
    
    for i, k in enumerate(keys):
        mask = masks_dict[k]
        sy0 = y0 - iy[i]
        sy1 = y1 - iy[i]
        sx0 = x0 - ix[i]
        sx1 = x1 - ix[i]
        aligned[k] = mask[sy0:sy1, sx0:sx1]

    return aligned



def get_region_properties(labeled_image):
    
    props = regionprops_table(labeled_image,
                              properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'area','label',))

    return pd.DataFrame(props)



def get_linkage_dict(label_props, min_distance, area_ratio, orientation_dif):
    
    timepoints = np.array(list(label_props.keys())).astype(int)
    
    connect_dict = {}
    lineage_dict = {}
    tr = 0
    
    for tm in range(np.min(timepoints), np.max(timepoints)):

        min_df = label_props[tm]
        max_df = label_props[tm+1]
        
        for ind, row in min_df.iterrows():
            
            x = row['centroid-1']
            y = row['centroid-0']
            ar = row.area
            ori = row.orientation
            cl_id = row.cell_id
            
            max_df['sd'] = np.sqrt((max_df['centroid-1']-x)**2+(max_df['centroid-0']-y)**2)
            max_df['ad'] = max_df.area / ar
            max_df['od'] = max_df.orientation - ori
            
            cls_df = max_df[(max_df.sd<min_distance)&
                            (max_df.ad.between(*area_ratio))&
                            (max_df.od.between(*orientation_dif))]
            if cls_df.shape[0]==1:
                connect_dict[cl_id] = cls_df.cell_id.values[0]
            elif cls_df.shape[0]>1:
                cls_df= cls_df[cls_df.sd==cls_df.sd.min()]
                connect_dict[cl_id] = cls_df.cell_id.values[0]
            elif cls_df.shape[0] == 0:
                connect_dict[cl_id] = 'none'
            
            if cl_id in lineage_dict:
                if connect_dict[cl_id]!='none':
                    lineage_dict[connect_dict[cl_id]] = lineage_dict[cl_id]
            elif cl_id not in lineage_dict:
                lineage_dict[cl_id] = 'traj_'+str(tr)
                tr+=1
                if connect_dict[cl_id]!='none':
                    lineage_dict[connect_dict[cl_id]] = lineage_dict[cl_id]
                
    return connect_dict, lineage_dict
            
        

def track_cells(unet_path, min_distance, area_ratio, orientation_dif):
    
    print('reading images...')
    masked_images = load_tif_images(unet_path)
    
    label_props = {}
    print('collecting label statistics...')
    for tmp in masked_images: 
         tmp_df = get_region_properties(masked_images[tmp])
         tmp_df['cell_id'] = tmp_df.label.astype('str')+'_'+str(tmp)
         tmp_df['frame'] = tmp
         label_props[tmp] = tmp_df

    print('connecting labels...')
    connect_dict, lineage_dict = get_linkage_dict(label_props, min_distance, area_ratio, orientation_dif)
    
    label_df = pd.concat(label_props, axis=0)
    label_df['link_id'] = label_df.cell_id.map(connect_dict)
    label_df['traj_id'] = label_df.cell_id.map(lineage_dict)
    
    size_dict = label_df.groupby('traj_id').frame.size().to_dict()
    label_df['traj_length']=  label_df.traj_id.map(size_dict)
    
    return label_df



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
    


def apply_cell_tracking(unet_path, experiment_id, min_distance=10, area_ratio=(0.95,1.1), orientation_dif=(-0.1,0.1),
                        min_trajectory_length=200, data_save_path='none', save_figure_path='none'):
    
    label_df = track_cells(unet_path, min_distance, area_ratio, orientation_dif)
    label_df['experiment_id'] = experiment_id[:experiment_id.find('_xy')]
    label_df['xy_position'] = int(experiment_id[experiment_id.find('_xy')+3:])
    print(label_df['experiment_id'].unique(), label_df['xy_position'].unique())
    if os.path.isdir(data_save_path):
        label_df.to_pickle(data_save_path+'/'+experiment_id+'_label_tracks_df', compression='zip')
    if os.path.isdir(save_figure_path):
        show_cell_trajectories(unet_path, label_df, min_trajectory_length, experiment_id, save_figure_path)
    
    
    
def track_multiple_xy_positions(folder_path, data_save_path):
    
    files = os.listdir(folder_path)
    files = [s for s in files if '_c1' in s]
    
    for fl in files:
        experiment_id = fl[:fl.find('_xy')]+fl[fl.find('_xy'):fl.find('_xy')+5]
        xy_position = int(fl[fl.find('_xy')+3:fl.find('_c1')])
        print(experiment_id, xy_position)
        
        omni_path = folder_path+'/'+fl+'/masks'
        
        apply_cell_tracking(omni_path, 
                            experiment_id, 
                            min_distance=10, 
                            area_ratio=(0.95,1.1), 
                            orientation_dif=(-0.1,0.1),
                            min_trajectory_length=200, 
                            data_save_path=data_save_path, 
                            save_figure_path='none')
        
        

def assemble_tracking_data(data_save_path):
    
    i = 0
    
    files = os.listdir(data_save_path)
    for fl in files:
        pos_df = pd.read_pickle(data_save_path+'/'+fl, compression='zip')
        if i == 0:
            final_df = pos_df
            i+=1
        else:
            final_df = pd.concat([final_df, pos_df])
            
    final_df.to_pickle(data_save_path+'/'+final_df.experiment_id.unique()[0]+'_all_tracks', compression='zip')
    
    return final_df
        
        
    
    
    