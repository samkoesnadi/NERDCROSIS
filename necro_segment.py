import numpy as np
import cv2 as cv
import scipy
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage.filters import threshold_otsu


def necro_segment(label, ori, threshold):
    ori = (ori/ori.max()*255).astype(np.uint8)
    label = (label/label.max()*255).astype(np.uint8)

    # plt.subplot(1, 2, 1)
    # plt.imshow(ori), plt.colorbar()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(label), plt.colorbar()
    #
    # plt.show()

    # threshold = threshold_otsu(ori/255)
    # print(threshold)

    pad = 50
    label = np.pad(label, (pad,), 'mean')
    label_first = label.copy()/label.max()
    ori = np.pad(ori, (pad,), 'mean')
    # # find label contour (the outter skin)
    # plt.imshow(label), plt.show()
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
    [vx,vy,x,y] = cv.fitLine(max_contour_outter, cv.DIST_L1, 0,0.01,0.01)
    #lefty = int((-x*vy/vx) + y)
    #righty = int(((cols-x)*vy/vx)+y)
    #print(vy,vx, (cols-1)/(righty-lefty))
    #cv.line(label,(cols-1,righty),(0,lefty),(0,255,0),2)
    # plt.imshow(label), plt.show()
    tetha = atan((vy/vx)[0])/pi*180
    tetha_fst = ((tetha+90)) if (tetha<0) else -(90-tetha) if (tetha>0) else 0
    #print(tetha_fst)
    #rotate it with tetha+90
    new_label = rotate(label, tetha_fst, clip=True)
    rotated_label = new_label.copy()
    new_ori = rotate(ori, tetha_fst, mode="edge",clip=True)
    rotated_ori = new_ori.copy()
    #plt.imshow(new_label), plt.show()
    # convert label to boolean data type (1 and 0)
    label = new_label.astype(np.uint8)
    label[label!=1] = 0
    label_bool = label.astype(np.bool_)
    label_int = label.astype(np.int_)

    # find label contour (the outter skin) for the new image (rotated)
    _, contours, _ = cv.findContours(label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rotated_max_contour_outter = 0
    for i, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        area_ref = cv.contourArea(contours[rotated_max_contour_outter])
        # print(area, area_ref)
        if (area > area_ref): rotated_max_contour_outter = i
    rotated_max_contour_outter = contours[rotated_max_contour_outter]

    # Region the new_ori and also label to roi_ori
    c = rotated_max_contour_outter
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    size = extRight[0]-extLeft[0]
    n = 10
    roi_label = label[extTop[1]-n:extTop[1]+size+n,extLeft[0]-n:extRight[0]+n]
    roi_ori = new_ori[extTop[1]-n:extTop[1]+size+n,extLeft[0]-n:extRight[0]+n]
    # plt.imshow(roi_label),plt.colorbar(),plt.show()
    # plt.imshow(roi_label),plt.colorbar(),plt.show()

    # OTSU Thresholding
    beta = 0.8
    from skimage.filters import threshold_otsu
    threshold2 = threshold_otsu(roi_ori)
    threshold = (1-beta)*threshold + beta*threshold2
    thresholded = roi_ori > threshold
    # plt.imshow(thresholded),plt.show()

    # # inner flesh
    how_much_deep = 8
    inner_flesh = cv.erode(roi_label, np.ones((10,10),np.uint8),iterations = how_much_deep)
    inner_flesh = thresholded | inner_flesh

    # Racing pixels procedure
    from racing_pixel import racing_pixel
    mask = roi_label.copy()

    _, contours, _ = cv.findContours(inner_flesh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rotated_max_contour = 0
    for i, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        area_ref = cv.contourArea(contours[rotated_max_contour])
        # print(area, area_ref)
        if (area > area_ref): rotated_max_contour = i
    rotated_max_contour = contours[rotated_max_contour]
    mask = cv.erode(mask, np.ones((5,5),np.uint8),iterations = 1)
    cv.drawContours(mask, [rotated_max_contour], -1, 2, -1)
    # plt.imshow(mask), plt.show()

    try:
        raced = racing_pixel(mask.copy(), n=1, alpha=0.9, beta=0.3, mode='constant', seperate_left_right=False)
    except:
        raced = mask
    raced_res = np.bitwise_and(raced, (1-thresholded))
    # plt.imshow(raced_res), plt.show()
    # distance_transform_edt
    distance = ndimage.distance_transform_edt(raced_res)
    # plt.imshow(distance), plt.colorbar(),plt.show()
    distance = distance/distance.max() if distance.max()!=0 else distance

    # plt.imshow(distance);plt.show()

    #Last
    distance_255 = (distance*255).astype(np.uint8)
    roi_ori = (roi_ori/roi_ori.max()*255)
    # plt.imshow(roi_ori+distance);plt.show()
    # ori_now = roi_ori+distance_255
    ori_now = roi_ori
    # plt.imshow(new_ori);plt.show()

    # rotate it back

    rotated_ori = rotated_ori/rotated_ori.max()*255
    rotated_label[extTop[1]-n:extTop[1]+size+n,extLeft[0]-n:extRight[0]+n] += distance
    rotated_ori[extTop[1]-n:extTop[1]+size+n,extLeft[0]-n:extRight[0]+n] = ori_now

    # # rotate it with tetha-90
    end_label = rotate(rotated_label, -tetha_fst)
    end_ori = rotate(rotated_ori, -tetha_fst, mode="edge")
    # end_label = rotated_label
    # end_ori = rotated_ori

    # crop weird stuffs
    end_label = end_label*label_first

    end_label = end_label[pad:-pad,pad:-pad]
    end_ori = end_ori[pad:-pad,pad:-pad]

    # plt.subplot(1, 2, 1)
    # plt.imshow(end_ori), plt.colorbar()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(end_label), plt.colorbar()
    #
    # plt.show()
    return end_label









# # GRAB Cut - Deprecated
# a = 0.7
# grabcut = np.zeros_like(roi_ori)
# grabcut[distance>0] = 2
# foo = np.bitwise_and(distance<=1,distance>=a)
# grabcut[foo] = 1
#
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
#
# img = cv.cvtColor((roi_ori/roi_ori.max()*255).astype(np.uint8),cv.COLOR_GRAY2RGB)
# grabcut = grabcut.astype(np.uint8)
#
# grabcut = binary_erosion(grabcut, iterations=3).astype(np.uint8)
# plt.imshow(grabcut), plt.colorbar(),plt.show()
# mask_grabcut, bgdModel, fgdModel = cv.grabCut(img,grabcut,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
# mask_grabcut = np.where((mask==1)|(mask==3),0,1).astype('uint8')
# img = img*mask_grabcut[:,:,np.newaxis]
#
# plt.imshow(img);plt.show()

