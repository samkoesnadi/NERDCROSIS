'''
Bound Box by Lukas. Editted by Samuel
'''
import cv2 as cv
import numpy as np

def findBoundBox(img): # input have to be between 0 - 255
    '''
    Argument:
    img = segmentation Pic
    Return:
    (i,j,k) = of respective left and right
    '''
    # print(img.shape)
    half_input_size = (img.shape[2])//2
    frames = img.shape[0]
    c_left = None # (x,y)_topleft, (x,y)_botright
    c_right = None # left, right, top, bot
    for j in range(frames):
        ret, thresh = cv.threshold(img[j], 10, 255,0)  # threshhold for contour detection (anything >= 0 ensures that everything that is not black is recognized)
        _, cnts, _ = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # determine the most extreme points along the contour

        for c in cnts:
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            # calculate moments for each contour
            M = cv.moments(c)

            # calculate x,y coordinate of center via moments
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if(cX <= half_input_size):
                    if (c_left==None):
                        c_left = ([0,0],[0,0],'l')
                        c_left[0][0] = extLeft[0]
                        c_left[0][1] = extTop[1]
                        c_left[1][0] = extRight[0]
                        c_left[1][1] = extBot[1]
                    c_left[0][0] = extLeft[0] if (c_left[0][0]>extLeft[0]) else c_left[0][0]
                    c_left[0][1] = extTop[1] if (c_left[0][1]>extTop[1]) else c_left[0][1]
                    c_left[1][0] = extRight[0] if (c_left[1][0]<extRight[0]) else c_left[1][0]
                    c_left[1][1] = extBot[1] if (c_left[1][1]<extBot[1]) else c_left[1][1]
                else:
                    if (c_right==None):
                        c_right = ([0,0],[0,0],'r')
                        c_right[0][0] = extLeft[0]
                        c_right[0][1] = extTop[1]
                        c_right[1][0] = extRight[1]
                        c_right[1][1] = extBot[1]
                    c_right[0][0] = extLeft[0] if (c_right[0][0]>extLeft[0]) else c_right[0][0]
                    c_right[0][1] = extTop[1] if (c_right[0][1]>extTop[1]) else c_right[0][1]
                    c_right[1][0] = extRight[0] if (c_right[1][0]<extRight[0]) else c_right[1][0]
                    c_right[1][1] = extBot[1] if (c_right[1][1]<extBot[1]) else c_right[1][1]
    # print(c_left,c_right)
    c_outputs = [c_left,c_right]
    return c_outputs

