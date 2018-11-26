import itk
import cv2 as cv
import numpy as np
import nibabel as nib

# itk_image = itk.imread('Datasets/P0001/coronal.gipl')
# itk_seg = itk.imread('Datasets/P0001/femurseg.gipl')

# itk_array = itk.GetArrayFromImage(itk_image) # here is the variable for segmented image
# seg = itk.GetArrayFromImage(itk_seg)
# seg[seg>0] = 1
# x_n = 512
# y_n = 255
# z = 48

# print(itk_array)
def interpolate_toNifti(itk_array, seg, x_n, y_n, z):
    seg[seg>0] = 1
    out = np.zeros((x_n,y_n,z)) # with nifti format
    out_seg = np.zeros((x_n,y_n,z))

    for i in range(itk_array.shape[0]):
        out[:,:,i] = np.flip(cv.resize(itk_array[i,:,:], dsize=(x_n, y_n), interpolation=cv.INTER_LINEAR).T,(0,1))
        out_seg[:,:,i] = np.flip(cv.resize(seg[i,:,:], dsize=(x_n, y_n), interpolation=cv.INTER_LINEAR).T,(0,1))
    #
    for i in range(x_n):
        # out[i,:,:] = np.flip(cv.resize(itk_array[:,:,i], dsize=(y_n, z), interpolation=cv.INTER_LINEAR).T,0)
        # out_seg[i,:,:] = np.flip(cv.resize(seg[:,:,i], dsize=(y_n, z), interpolation=cv.INTER_LINEAR).T,0)
        out[i,:,:] = (cv.resize(out[:,:,:itk_array.shape[0]][i,:,:], dsize=(z, y_n), interpolation=cv.INTER_LINEAR))
        out_seg[i,:,:] = (cv.resize(out_seg[:,:,:itk_array.shape[0]][i,:,:], dsize=(z, y_n), interpolation=cv.INTER_LINEAR))
    #
    out_seg[out_seg>0] = 1
    return (out, out_seg)

# outNifti = nib.Nifti1Image(out, affine=np.eye(4))
# outNifti_seg = nib.Nifti1Image(out_seg, affine=np.eye(4))
# # print(out.shape)
# nib.save(outNifti, 'out.nii.gz')
# nib.save(outNifti_seg, 'out_seg.nii.gz')
# # itk.imwrite(out,'out.gipl')

