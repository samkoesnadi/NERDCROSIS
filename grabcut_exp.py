# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('sample_2ds/0.png', 0)
# shape = img.shape
# img = np.pad(img, (50,), 'mean')
# # mask = np.zeros(img.shape[:2],np.uint8)
# # bgdModel = np.zeros((1,65),np.float64)
# # fgdModel = np.zeros((1,65),np.float64)
# rect = (0,0,shape[1],shape[0 ])
# # cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
# # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# # img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

# -*- coding: utf-8 -*-
import numpy as np
#import cv2 as cv
import cv2
from matplotlib import pyplot as plt

import numpy as np
import cv2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#Load the Image

imgo = cv2.imread('sample_2ds/0.png')
imgo = cv2.resize(imgo, (300, 500))
cv2.imshow('IMAGE',imgo)

height, width = imgo.shape[:2]

#Create a mask holder
mask = np.zeros(imgo.shape[:2],np.uint8)

#Grab Cut the object
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#  the Rect… The object must lie within this rect.
rect = (25,25,width-20,height-20)
#rect = (47,164,92,80) # why it doesnt work!!

cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==1)|(mask==3),0,1).astype('uint8')
img1 = imgo*mask[:,:,np.newaxis]

#Get the background
background = imgo - img1

#Change all pixels in the background that are not black to white
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

#Add the background and the image
final = background + img1

# mb later on, Erode
cv2.imshow('image', final )

k = cv2.waitKey(0)
