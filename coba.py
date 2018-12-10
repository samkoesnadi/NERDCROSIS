import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # for visualising and debugging
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries

# read the image
label = cv.imread("sample_2ds/2_Label.png",0)
ori = cv.imread("sample_2ds/2.png",0)

pad = 50
label = np.pad(label, (pad,), 'mean')
ori = np.pad(ori, (pad,), 'mean')
# # find label contour (the outter skin)
_, contours, _ = cv.findContours(label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_contour_outter = 0
for i, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    area_ref = cv.contourArea(contours[max_contour_outter])
    # print(area, area_ref)
    if (area > area_ref): max_contour_outter = i
max_contour_outter = contours[max_contour_outter]

# find line to check the rotation angle
from math import atan,pi
from skimage.transform import rotate
rows,cols = label.shape[:2]
[vx,vy,x,y] = cv.fitLine(max_contour_outter, cv.DIST_L2, 0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(label,(cols-1,righty),(0,lefty),(0,255,0),2)
# plt.imshow(label), plt.show()
theta = -atan(vy/vx)-pi/2
print(theta)
#rotate it with tetha+90
# new_label = rotate(label, tetha+90,clip=True)
from skimage.transform import AffineTransform, SimilarityTransform, warp, EuclideanTransform
shift_y, shift_x = np.array(label.shape[:2]) / 2.
tf_rotate = SimilarityTransform(rotation=theta)
tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

new_label = warp(label, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
plt.imshow(new_label), plt.show()
