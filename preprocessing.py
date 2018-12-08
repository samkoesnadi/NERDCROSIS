'''
GIPL to GIPL

Lukas and Samuel 5.11.2018
'''

import itk
import numpy as np
import cv2 as cv
from math import floor, ceil
from os import listdir, makedirs
from os.path import isfile, join, exists
from re import search
from bound_box import findBoundBox
from interpolate import interpolate
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries

W_0, SIGMA = 10, 5

directory_of_datasets = "./Datasets" #datasets folder
picture_save_dir = 'data'
output_dir_0 = join('./niftynet/', picture_save_dir)
ori_image_filename = "coronal.gipl"
onlyfiles = ["femurseg.gipl"]# this is the files names that you want to extract
postfix_ori_out = "Coronal"
postfix_seg_out = "Label"
which_patients = listdir(directory_of_datasets)
exclude_patients = ["P0011","P0013","P0018"]

z = 48
n = 512 #Side length of square image
pad_out_percentage = 20

if __name__ == '__main__':
    file_index = 0
    # which_patients = ["P0032"]
    for which_patient in which_patients:
        if (which_patient in exclude_patients): continue
        dir_patient = join(directory_of_datasets,which_patient)
        output_dir = join(output_dir_0, which_patient)
        for fi in onlyfiles: # fi is the file that we want to convert, e.g. t1_tse_cor.gipl
            temp_out_dir = (output_dir)

            if (not exists(join(dir_patient,fi))): print('There is no',join(dir_patient,fi)); continue
            if (not exists(join(dir_patient, ori_image_filename))): print('There is no',join(dir_patient, ori_image_filename)); continue
            print('Analyzing',(dir_patient))
            itk_image = itk.imread(join(dir_patient,fi))
            ori_image = itk.imread(join(dir_patient, ori_image_filename))# this is the original MRI Pic


            # Copy of itk.Image, data is copied
            segment = itk.GetArrayFromImage(itk_image) # here is the variable for segmented image
            pad_ = int(pad_out_percentage/100*(segment.shape[0]//4))
            segment = np.pad(segment,(pad_),'minimum')
            segment_255 = np.interp(segment, (segment.min(), segment.max()), (0, 255))
            segment_255 = segment_255.astype(np.uint8)


            ori_arr = itk.GetArrayFromImage(ori_image)
            ori, segment = interpolate(ori_arr, segment, 560, 560, 224)

            segment = segment.astype(np.uint8)
            ori = ori.astype(np.float32)

            new_segment = itk.GetImageFromArray(segment)
            new_ori = itk.GetImageFromArray(ori)

            dimension = new_ori.GetImageDimension()
            InputImageType = type(new_ori)
            OutputImageType = type(itk_image)

            #casting oti
            cast_new_ori = itk.CastImageFilter[InputImageType, OutputImageType].New(new_ori)

            print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.gipl'),'from',join(dir_patient,fi))
            itk.imwrite(cast_new_ori, join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.gipl'))
            print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.gipl'))
            itk.imwrite(new_segment, join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.gipl'))
            file_index += 1
