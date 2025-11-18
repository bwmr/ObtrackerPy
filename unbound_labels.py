# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:22:23 2025

@author: Alexandros Papagiannakis, HHMI @Stanford University, 2025
"""

import numpy as np


def remove_cells_at_boundaries(image_labels):
    
    height = image_labels.shape[0]
    width = image_labels.shape[1]
    
    bound_labels = list(image_labels[0,:].ravel())
    bound_labels += list(image_labels[:,0].ravel())
    bound_labels += list(image_labels[height-1,:].ravel())
    bound_labels += list(image_labels[:,width-1].ravel())
    bound_labels = np.unique(bound_labels)    
    
    index_to_delete = np.where(bound_labels == 0)
    bound_labels = np.delete(bound_labels, index_to_delete)
    
    print(f'labels at boundaries: {bound_labels}')
    image_labels[np.isin(image_labels, bound_labels)]=0
    
    return image_labels



def apply_boundary_removal(image_labels_dict):
    
    unbound_labels = {}
    
    keys = sorted(image_labels_dict.keys())
    for i, k in enumerate(keys):
        print(f'frame: {k}')
        unbound_labels[k] = remove_cells_at_boundaries(image_labels_dict[k])
    return unbound_labels
        