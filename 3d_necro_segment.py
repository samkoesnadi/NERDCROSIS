import numpy as np
import cv2 as cv
import itk
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
# from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from scipy import ndimage

# read the image
oris_itk = itk.imread("sample_2ds/coronal.gipl",0)
labels_itk = itk.imread("sample_2ds/femurseg.gipl",0)
oris = itk.GetArrayFromImage(oris_itk)
labels = itk.GetArrayFromImage(labels_itk)
labels[labels>1] = 1

pad = 50
labels = np.pad(labels, (pad,), 'mean')
oris = np.pad(oris, (pad,), 'mean')

# first locate the bones using dbscan! CLUSTERING
