"""
Created on Mon Nov 11 21:34:14 2024.

@author: Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""


from pathlib import Path

import pandas as pd
from tqdm import tqdm

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

    if cum_drift is not None:
         print('applying drift correction...')
         masked_images = dc.apply_drift_correction(masked_images,
                                                   cum_drift[0],
                                                   cum_drift[1])

    print('removing cells at boundaries...')
    masked_images = ul.apply_boundary_removal(masked_images)

    print('collecting label statistics...')
    label_props = {}
    for tmp in tqdm(masked_images):
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
                        do_drift = False,
                        search_radius=10,
                        area_ratio=(0.9,1.2),
                        orientation_dif=(-0.1,0.1),
                        diag_plot = True,
                        min_trajectory_length=100):
    """Apply cell tracking to experiment.

    Inputs:
        mask_path: path containing omnipose-derived mask files
        experiment_id: unique ID for each experiment and condition
        do_drift: if drifts were calculated using UnDrift, apply those drifts
        search_radius: maximal centroid distance to consider for lineage tracing
        area_ratio: consider [upper, lower]-fold area change between images
        orientation_dif: consider this change to orientation of medial axis (rad)
        diag_plot: output a diagnostic area over time plot
        min_trajectory_length: plot only trajectories of at least this length

    Returns:
        Outputs diagnostic plot for area over time for each trajectory.
        Saves label tracks to mask parent folder.

    """
    cum_drift = None

    unet_path = Path(unet_path)

    if do_drift:
        drift_save_path = str(unet_path.parent)

        _, cum_drift = dc.load_drift_statistics(drift_save_path)

    label_df = track_cells(unet_path,
                           cum_drift,
                           search_radius,
                           area_ratio,
                           orientation_dif)

    label_df['experiment_id'] = experiment_id[:experiment_id.find('_xy')]

    label_df['xy_position'] = int(experiment_id[experiment_id.find('_xy')+
                                                3:experiment_id.find('_xy')+5])

    print(f'for {label_df["experiment_id"].unique()[0]}, there are '
          f'{label_df["xy_position"].unique()[0]} tracks.')

    label_df.to_pickle(unet_path.parent / f'{experiment_id}_label_tracks.pkl',
                       compression='zip')

    if diag_plot:
        viz.show_cell_trajectories(label_df, min_trajectory_length)


