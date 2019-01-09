import numpy as np
import cv2 as cv
import itk
import time
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
# from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from sklearn.cluster import DBSCAN

def d_necro_seg(oris, labels):
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

    # # take first label as experiment
    # find line to check the rotation angle
    labels = labels/labels.max()
    labels_res = np.zeros_like(labels)
    a = 0
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

        end_label_3d = np.zeros_like(label_ROI)
        index = 0
        for ori, label in zip(ori_ROI,label_ROI):
            end_label = label

            # if you want to digitalize the value than uncomment this
            end_label = np.ceil(end_label)
            
            end_label_3d[index] = (end_label)
            index += 1
        labels_res[z_axis0.min():z_axis0.max()+1,y_axis0.min():y_axis0.max()+1,x_axis0.min():x_axis0.max()+1] = end_label_3d

    print("--- %s seconds ---" % (time.time() - start_time))

    # just so that you can see in plot clearly ---->
    # labels_res = labels_res/labels_res.max()*255

    # for ori, label in zip(oris,labels_res):
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(ori), plt.colorbar()
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(label), plt.colorbar()
    #
    #     plt.show()

    return labels_res


# read the image
from os import listdir
from os.path import isfile, join
mypath = 'niftynet/temp'
oris_path = "niftynet/data/"
segment_filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(segment_filenames)

for segment_filename in segment_filenames:
    if (segment_filename[-11:]!='_Label.gipl'): continue
    # if(segment_filename!='0_niftynet_out.nii.gz'): continue
    prefix = segment_filename[0:-10]
    oris_file_path = join(oris_path, prefix+'Coronal.gipl')
    segment_file_path = join(mypath, segment_filename)
    print('\n\nAnalyzing: ',oris_file_path,'\n')
    img = itk.imread(segment_file_path, 0)
    labels = itk.GetArrayFromImage(img)
    oris_itk = itk.imread(oris_file_path, 0)
    oris = itk.GetArrayFromImage(oris_itk)

    segment = d_necro_seg(oris, labels)

    # segment = segment.astype(np.float32)
    segment = segment.astype(np.uint8)

    new_segment = itk.GetImageFromArray(segment)

    # dimension = new_segment.GetImageDimension()
    # InputImageType = type(new_segment)
    # OutputImageType = type(oris_itk)
    # cast_new_segment = itk.CastImageFilter[InputImageType, OutputImageType].New(new_segment)

    print("Writing neucro segment in ",join(mypath, prefix+"_necro_segment.gipl"))
    itk.imwrite(new_segment, join(mypath, prefix+"_necro_segment.gipl"))

