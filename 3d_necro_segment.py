import numpy as np
import cv2 as cv
import itk
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
# from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

# read the image
from os import listdir
from os.path import isfile, join
mypath = 
segment_filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

oris_itk = itk.imread("niftynet/data/2_Coronal.gipl", 0)
labels_itk = itk.imread("niftynet/data/2_Label.gipl", 0)
oris = itk.GetArrayFromImage(oris_itk)
labels = itk.GetArrayFromImage(labels_itk)

# start time counter
import time
start_time = time.time()

# first locate the bones using dbscan! CLUSTERING
from sklearn.cluster import DBSCAN

# change to 3D coordinate space [x,y,z] (check if labels==1)
## logical threshold = 119187 // 2 - n
one_pixel_neighbours = 200
distance_sample = 5

labels_flat = np.where(labels>0)

# labels_space in form of [z,y,x]
labels_space = np.dstack(labels_flat)[0]

# result = np.zeros_like(labels)
db = DBSCAN(eps=distance_sample, min_samples=one_pixel_neighbours).fit(labels_space)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_db = db.labels_
# print(db.core_sample_indices_.shape)
# print(db.components_.shape)

# Black removed and is used for noise instead.
unique_labels = set(labels_db)

# save it to labels_ready in [x,y,z] format
labels_ready = []
for k in zip(unique_labels):

    class_member_mask = (labels_db == k)

    xy = labels_space[class_member_mask & core_samples_mask]
    labels_ready.append(xy)
    # xy = labels_space[class_member_mask & ~core_samples_mask] # first we can ignore the non-core samples
labels_ready = np.array(labels_ready)

# # take first label as experiment
# find line to check the rotation angle
labels = labels/labels.max()
for i, label_ready in enumerate(labels_ready):

    # now output the ROI
    x_axis0 = label_ready[...,2]
    y_axis0 = label_ready[...,1]
    z_axis0 = label_ready[...,0]
    x_axis = x_axis0 - x_axis0.min()
    y_axis = y_axis0 - y_axis0.min()
    z_axis = z_axis0 - z_axis0.min()
    label_ROI = np.zeros((z_axis.max()+1,y_axis.max()+1,x_axis.max()+1))
    label_ROI[z_axis,y_axis,x_axis] = 1 # draw one to respective label

    ori_ROI = oris[z_axis0.min():z_axis0.max()+1,y_axis0.min():y_axis0.max()+1,x_axis0.min():x_axis0.max()+1]


    # SKIP THE 3-D ROTATION TRANSFORM (TO-DO)

    # continue with layer per layer neucrosis segmentation
    from skimage.filters import threshold_otsu
    threshold = threshold_otsu(ori_ROI/ori_ROI.max())

    from necro_segment import necro_segment
    end_label_3d = np.zeros_like(label_ROI)
    index = 0
    for ori, label in zip(ori_ROI,label_ROI):
        end_label = necro_segment(label, ori, threshold)
        end_label[end_label>0] += index
        end_label_3d[index] = (end_label)
        index += 1
    labels[z_axis0.min():z_axis0.max()+1,y_axis0.min():y_axis0.max()+1,x_axis0.min():x_axis0.max()+1] = end_label_3d

print("--- %s seconds ---" % (time.time() - start_time))

# just so that you can see in plot clearly ---->
labels = labels/labels.max()*255

for ori, label in zip(oris,labels):
    plt.subplot(1, 2, 1)
    plt.imshow(ori), plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(label), plt.colorbar()

    plt.show()
