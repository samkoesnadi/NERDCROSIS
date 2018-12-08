import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # for visualising and debugging
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries

W_0, SIGMA = 10, 5


def construct_weights_and_mask(imgs):
    weight_maps = []
    for img in imgs:
        seg_boundaries = find_boundaries(img, mode='inner')
        # print(seg_boundaries)
        weight_map = np.ones_like(seg_boundaries)*W_0
        cell_ids = [x for x in np.unique(img) if x > 0]
        print(np.any(seg_boundaries), len(cell_ids))
        if (np.any(seg_boundaries) and len(cell_ids)>1):
            print("inside")
            bin_img = img > 0
            # take segmentations, ignore boundaries
            binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

            foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
            background_weight = 1 - foreground_weight

            # build euclidean distances maps for each cell:
            distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

            for i, cell_id in enumerate(cell_ids):
                distances[..., i] = distance_transform_edt(img != cell_id)

            # we need to look at the two smallest distances
            distances.sort(axis=-1)

            if len(cell_ids) != 0:
                weight_map *= np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))

                weight_map[binary_with_borders] = foreground_weight
                weight_map[~binary_with_borders] += background_weight
        print(weight_map.shape)
        weight_maps.append([weight_map])
    return np.array(weight_maps)
    
def construct_weights_and_mask(img):
    seg_boundaries = find_boundaries(img, mode='inner')

    bin_img = img > 0
    # take segmentations, ignore boundaries
    binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

    foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
    background_weight = 1 - foreground_weight

    # build euclidean distances maps for each cell:
    cell_ids = [x for x in np.unique(img) if x > 0]
    distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        distances[..., i] = distance_transform_edt(img != cell_id)

    # we need to look at the two smallest distances
    distances.sort(axis=-1)

    weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
    weight_map[binary_with_borders] = foreground_weight
    weight_map[~binary_with_borders] += background_weight

    return weight_map, binary_with_borders

plt.imshow(construct_weights_and_mask(cv.imread("sample_2ds/4_Label.png",0))[0]), plt.show()
