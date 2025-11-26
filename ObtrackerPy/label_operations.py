"""
Created on Mon Nov 17 22:23:29 2025

@author: Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""


import os

import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops_table


def load_tif_images(unet_path, time_string_length=4, image_suffix_str='_cp_masks'):

    phase_objects = os.listdir(unet_path)
    masks_paths = [s for s in phase_objects if 'cp_masks' in s]

    masked_images = {}

    for msk in masks_paths:
        tm = msk[msk.find(image_suffix_str)-time_string_length:msk.find(image_suffix_str)]
        print('reading image timepoint:', tm)
        masked_images[int(tm)] = io.imread(unet_path+'/'+msk)

    return masked_images



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
