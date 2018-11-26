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
import sys

directory_of_datasets = "./Datasets" #datasets folder
picture_save_dir = 'dense_vnet_femur'
output_dir_0 = join('./niftynet/data', picture_save_dir)
ori_image_filename = "coronal.gipl"
onlyfiles = ["femurseg.gipl"]# this is the files names that you want to extract
postfix_ori_out = "Coronal"
postfix_seg_out = "Label"
which_patients = listdir(directory_of_datasets)

z = 48
n = 512 #Side length of square image
pad_ = ((0, 0), (n//8, n//8), (n//8, n//8))
pad_out_percent = 6

size_end = (256, 256, 48)

if __name__ == '__main__':
    file_index = 0
    #which_patients = ["P0017"]
    for which_patient in which_patients:
        

        dir_patient = join(directory_of_datasets,which_patient)
        output_dir = join(output_dir_0, which_patient)
        for fi in onlyfiles: # fi is the file that we want to convert, e.g. t1_tse_cor.gipl
            temp_out_dir = (output_dir)

            if (not exists(join(dir_patient,fi))): print('There is no',join(dir_patient,fi)); continue
            if (not exists(join(dir_patient, ori_image_filename))): print('There is no',join(dir_patient, ori_image_filename)); continue
            print('--- Analyzing ---',(dir_patient))

            itk_image = itk.imread(join(dir_patient,fi))
            ori_image = itk.imread(join(dir_patient, ori_image_filename))# this is the original MRI Pic

            # Copy of itk.Image, data is copied
            segment = itk.GetArrayFromImage(itk_image) # here is the variable for segmented image
            segment = np.pad(segment,(pad_),'minimum')
            segment_255 = np.interp(segment, (segment.min(), segment.max()), (0, 255))
            segment_255 = segment_255.astype(np.uint8)


            ori_arr = itk.GetArrayFromImage(ori_image)

            ori_shape = ori_arr.shape
            ori_arr = np.pad(ori_arr,(pad_),'edge')
            try:
                for c in findBoundBox(segment_255):
                    # y_n = c[1][1]-(c[0][1]-pad_out*4)
                    # x_n = c[1][0]+pad_out*4-(c[0][0]-pad_out*4)
                    y_n = ori_shape[1]//2
                    x_n = ori_shape[2]//2; pad_out = int(pad_out_percent/100*y_n)
                    ins = np.zeros((segment.shape[0],y_n,x_n))
                    in_seg = np.zeros((segment.shape[0],y_n,x_n))

                    for i in range(segment.shape[0]):
                        # out_raw = ori_arr[i, c[0][1]-pad_out*4:c[1][1], c[0][0]-pad_out*4:c[1][0]+pad_out*4]
                        # out_raw_seg = segment[i, c[0][1]-pad_out*4:c[1][1], c[0][0]-pad_out*4:c[1][0]+pad_out*4]

                        if (c[2]=='l'):
                            #print(c,ori_arr.shape,i, c[0][1]-pad_out,c[0][1]-pad_out+y_n, c[0][0]-pad_out,c[0][0]-pad_out+x_n)
                            out_raw = ori_arr[i, c[0][1]-pad_out:c[0][1]-pad_out+y_n, c[0][0]-pad_out:c[0][0]-pad_out+x_n]
                            out_raw_seg = segment[i, c[0][1]-pad_out:c[0][1]-pad_out+y_n, c[0][0]-pad_out:c[0][0]-pad_out+x_n]
                            out_finish = out_raw
                        else:
                            #print(c,ori_arr.shape,i, c[0][1]-pad_out,c[0][1]-pad_out+y_n, c[1][0]-pad_out-x_n,c[1][0]-pad_out)
                            out_raw = ori_arr[i, c[0][1]-pad_out:c[0][1]-pad_out+y_n, c[1][0]+pad_out-x_n:c[1][0]+pad_out]
                            out_raw_seg = segment[i, c[0][1]-pad_out:c[0][1]-pad_out+y_n, c[1][0]+pad_out-x_n:c[1][0]+pad_out]

                            out_finish = np.flip(out_raw,1)
                            out_raw_seg = np.flip(out_raw_seg,1)
                        ins[i] = out_finish
                        in_seg[i] = out_raw_seg
                    (out, out_seg) = interpolate_toNifti(ins,in_seg,size_end[0],size_end[1],size_end[2])
                    out = out.astype(np.int32); out_seg = out_seg.astype(np.int32)
                    outNifti = nib.Nifti1Image(out, affine=np.eye(4))
                    outNifti_seg = nib.Nifti1Image(out_seg, affine=np.eye(4))

                    print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.nii.gz'),'from',join(dir_patient,fi))
                    nib.save(outNifti, join(output_dir_0,str(file_index)+'_'+postfix_ori_out+'.nii.gz'))
                    print("Writing",join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.nii.gz'))
                    nib.save(outNifti_seg, join(output_dir_0,str(file_index)+'_'+postfix_seg_out+'.nii.gz'))
                    file_index += 1
            except ValueError as e:
                print(e)
            except:
                print("Unexpected error:", sys.exc_info()[0])

