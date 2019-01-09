'''
GIPL to GIPL

Lukas and Samuel 5.11.2018
'''

import itk
import nibabel as nib
import numpy as np
import uuid
import cv2 as cv
from math import floor, ceil
from os import listdir, makedirs
from os.path import isfile, join, exists
from re import search
from bound_box import findBoundBox
from interpolate import interpolate_toNifti

directory_of_datasets = "./Datasets" #datasets folder
picture_save_dir = 'data'
output_dir_0 = join('./niftynet/', picture_save_dir)
ori_image_filename = "coronal.gipl"
onlyfiles = ["necro_seg.gipl"]# this is the files names that you want to extract
postfix_ori_out = "Coronal"
postfix_seg_out = "Label"
which_patients = listdir(directory_of_datasets)

#z = 48
#n = 512 #Side length of square image
pad_out_percentage = 50

if __name__ == '__main__':
    file_index = 0
    which_patients = ["P0001"]
    for which_patient in which_patients:
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
            pad_ = int(pad_out_percentage/100*(segment.shape[2]//4))
            npad = ((0,0),(pad_, pad_),(pad_, pad_))

            segment = np.pad(segment, pad_width=npad, mode='minimum')
            segment_255 = np.interp(segment, (segment.min(), segment.max()), (0, 255))
            segment_255 = segment_255.astype(np.uint8)


            ori_arr = itk.GetArrayFromImage(ori_image)
            ori_arr = np.pad(ori_arr, pad_width=npad, mode='edge')
            
            for c in findBoundBox(segment_255):
                y_n = segment_255.shape[1]
                x_n = c[1][0]+pad_-(c[0][0]-pad_)
                ins = np.zeros((segment.shape[0],y_n,x_n))
                in_seg = np.zeros((segment.shape[0],y_n,x_n))
                print(x_n,y_n,segment.shape[0])
                for i in range(segment.shape[0]):
                    out_raw = ori_arr[i, 0:y_n, c[0][0]-pad_:c[1][0]+pad_]
                    out_raw_seg = segment[i, 0:y_n, c[0][0]-pad_:c[1][0]+pad_]
                    if (c[2]=='l'):
                        out_finish = out_raw
                    else:
                        #out_finish = np.flip(out_raw,1)
                        #out_raw_seg = np.flip(out_raw_seg,1)
                        out_finish = out_raw

                    ins[i] = out_finish
                    in_seg[i] = out_raw_seg
                (out, out_seg) = interpolate_toNifti(ins,in_seg,x_n,y_n,segment_255[0])

                outNifti = nib.Nifti1Image(out, affine=np.eye(4))
                outNifti_seg = nib.Nifti1Image(out_seg, affine=np.eye(4))

                print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.nii.gz'),'from',join(dir_patient,fi))
                nib.save(outNifti, join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.nii.gz'))
                print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.nii.gz'))
                nib.save(outNifti_seg, join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.nii.gz'))
                file_index += 1
