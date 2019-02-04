# to process the image after getting ti from the first network... make the boundary of it
# output is resized 60x60

import numpy as np
import cv2 as cv
import itk
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from sklearn.cluster import DBSCAN

index = 0

def d_necro_seg(oris, labels, necros):
    # start time counter
    start_time = time.time()

    # first locate the bones using dbscan! CLUSTERING


    # change to 3D coordinate space [x,y,z] (check if labels==1)
    ## logical threshold = 119187 // 2 - n
    one_pixel_neighbours = 500
    distance_sample = 15

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
    print('Labels detected in input images: ',unique_labels)
    if (-1 in unique_labels): unique_labels.remove(-1)
    for k in zip(unique_labels):

        class_member_mask = (labels_db == k)

        xy = labels_space[class_member_mask & core_samples_mask]
        labels_ready.append(xy)
        # xy = labels_space[class_member_mask & ~core_samples_mask] # first we can ignore the non-core samples
    labels_ready = np.array(labels_ready)

    for i, label_ready in enumerate(labels_ready):

        # now output the ROI
        x_axis0 = label_ready[...,2]
        y_axis0 = label_ready[...,1]
        z_axis0 = label_ready[...,0]
        x_axis = x_axis0 - x_axis0.min()
        y_axis = y_axis0 - y_axis0.min()
        z_axis = z_axis0 - z_axis0.min()
        label_ROI = np.zeros((z_axis.max()+1,y_axis.max()+1,x_axis.max()+1))
        necro_ROI = np.zeros((z_axis.max()+1,y_axis.max()+1,x_axis.max()+1))
        label_ROI[z_axis,y_axis,x_axis] = 1 # draw one to respective label
        necro_ROI[z_axis,y_axis,x_axis] = necros[z_axis0,y_axis0,x_axis0]
        ori_ROI = oris[z_axis0.min():z_axis0.max()+1,y_axis0.min():y_axis0.max()+1,x_axis0.min():x_axis0.max()+1]
        
        # mask the ori with the femur head label
        ori_ROI[np.where(label_ROI==0)] = 0
        a = []
        for ori, label, necro in zip(ori_ROI, label_ROI, necro_ROI):
            ori = cv.resize(ori, (60, 60)) 
            label = cv.resize(label, (60, 60))
            necro = cv.resize(necro, (60, 60))
            a.append([ori, label, necro])
        np.save('npy_middle/'+str(index)+'_'+str(i)+'.npy',np.array(a))
        


    print("--- %s seconds ---" % (time.time() - start_time))

    return (x_axis0.min(), y_axis0.min(), z_axis0.min(), ori_ROI.shape[2], ori_ROI.shape[1])




# read the image
from os import listdir
from os.path import isfile, join
mypath = 'niftynet/data_middle'
oris_path = "niftynet/data_raw/"
segment_filenames = sorted([f for f in listdir(oris_path) if isfile(join(oris_path, f))])

print(segment_filenames)

#segment_filenames = ["6_fLabel.gipl"]

f = open('npy_middle/positions.txt', 'w')
f.truncate(0)

for segment_filename in segment_filenames:
    if (segment_filename[-12:]!='_fLabel.gipl'): continue
    # if(segment_filename!='0_niftynet_out.nii.gz'): continue
    prefix = segment_filename[0:-11]
    oris_file_path = join(oris_path, prefix+'Coronal.gipl')
    segment_file_path = join(oris_path, segment_filename)
    necrosis_file_path = join(oris_path, prefix+'nLabel.gipl')
    print('\n\nAnalyzing: ',oris_file_path,'\n')
    img = itk.imread(segment_file_path, 0)
    heads = itk.GetArrayFromImage(img)
    oris = itk.GetArrayFromImage(itk.imread(oris_file_path, 0))
    necros = itk.GetArrayFromImage(itk.imread(necrosis_file_path, 0))
    
    # annotate to positions.txt
    f.write(';'+str(index)+','+str(prefix)+'\n')

    # boundary min value
    (x_axis, y_axis, z_axis, width, height) = d_necro_seg(oris, heads, necros)
    f.write(str(x_axis)+','+str(y_axis)+','+str(z_axis)+','+str(width)+','+str(height)+'\n')
    index += 1

f.close()
