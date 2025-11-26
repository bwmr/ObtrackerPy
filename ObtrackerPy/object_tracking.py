"""
Created on Mon Nov 11 21:34:14 2024.

@author: Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""


from pathlib import Path

import pandas as pd

from . import drift_correction as dc
from . import label_operations as lo
from . import unbound_labels as ul
from . import visualize_tracks as viz


def track_cells(unet_path,
                cum_drift,
                min_distance,
                area_ratio,
                orientation_dif):
    """Track cells between segmented images."""
    print('reading images...')
    masked_images = lo.load_tif_images(unet_path)

    print('applying drift correction...')
    masked_images = dc.apply_drift_correction(masked_images,
                                              cum_drift[0],
                                              cum_drift[1])

    print('removing cells at boundaries...')
    masked_images = ul.apply_boundary_removal(masked_images)

    print('collecting label statistics...')
    label_props = {}
    for tmp in masked_images:
         tmp_df = lo.get_region_properties(masked_images[tmp])
         tmp_df['cell_id'] = tmp_df.label.astype('str')+'_'+str(tmp)
         tmp_df['frame'] = tmp
         label_props[tmp] = tmp_df

    print('connecting labels...')
    connect_dict, lineage_dict = lo.get_linkage_dict(label_props,
                                                     min_distance,
                                                     area_ratio,
                                                     orientation_dif)

    label_df = pd.concat(label_props, axis=0)
    label_df['link_id'] = label_df.cell_id.map(connect_dict)
    label_df['traj_id'] = label_df.cell_id.map(lineage_dict)

    size_dict = label_df.groupby('traj_id').frame.size().to_dict()
    label_df['traj_length']=  label_df.traj_id.map(size_dict)

    return label_df



def apply_cell_tracking(unet_path,
                        experiment_id,
                        min_distance=10,
                        area_ratio=(0.95,1.1),
                        orientation_dif=(-0.1,0.1),
                        min_trajectory_length=100):
    """Apply cell tracking to experiment."""
    drift_save_path = str(Path(unet_path).parent)
    _, cum_drift = dc.load_drift_statistics(drift_save_path)

    label_df = track_cells(unet_path,
                           cum_drift,
                           min_distance,
                           area_ratio,
                           orientation_dif)

    label_df['experiment_id'] = experiment_id[:experiment_id.find('_xy')]

    label_df['xy_position'] = int(experiment_id[experiment_id.find('_xy')+
                                                3:experiment_id.find('_xy')+5])

    print(label_df['experiment_id'].unique(), label_df['xy_position'].unique())

    label_df.to_pickle(drift_save_path+'/'+experiment_id+'_label_tracks_df',
                       compression='zip')

    viz.show_cell_trajectories(label_df, min_trajectory_length)


