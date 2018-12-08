import numpy as np
import cv2 as cv
import scipy
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import ndimage

# W_0, SIGMA = 10, 5
#
#
# def construct_weights_and_mask(img):
#     seg_boundaries = find_boundaries(img, mode='inner')
#
#     bin_img = img > 0
#     # take segmentations, ignore boundaries
#     binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)
#
#     foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
#     background_weight = 1 - foreground_weight
#
#     # build euclidean distances maps for each cell:
#
#     cell_ids = [x for x in np.unique(img) if x > 0]
#     distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))
#
#     for i, cell_id in enumerate(cell_ids):
#         distances[..., i] = distance_transform_edt(img != cell_id)
#         # (distances[..., i]) = distances[..., i].astype(np.uint8)
#         # cv.imshow('result',distances[..., i])
#         #
#         # cv.waitKey(0)
#
#     # we need to look at the two smallest distances
#     distances.sort(axis=-1)
#
#     weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
#     weight_map[binary_with_borders] = foreground_weight
#     weight_map[~binary_with_borders] += background_weight
#
#     return weight_map, binary_with_borders

label = cv.imread("sample_2ds/1_Label.png",0)
ori = cv.imread("sample_2ds/1.png",0)

# label = cv.resize(label, (300, 500))
# ori = cv.resize(ori, (300, 500))

source1 = ori.copy()
# ori[label<=1] = 0
# label = label.astype(np.uint8)
# label *= 255

outter = label.copy()
# kernel = np.ones((3,3),np.uint8)
# outter = cv.erode(outter,kernel,iterations = 20)
_, contours, _ = cv.findContours(outter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_contour_outter = 0
for i, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    area_ref = cv.contourArea(contours[max_contour_outter])
    # print(area, area_ref)
    if (area > area_ref): max_contour_outter = i
max_contour_outter = contours[max_contour_outter]

# determine the most extreme points along the contour
# c = max_contour_outter
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])
#
# size = extRight[0]-extLeft[0]
# label2 = label[extTop[1]:extTop[1]+size,extLeft[0]:extRight[0]]
# ori2 = ori[extTop[1]:extTop[1]+size,extLeft[0]:extRight[0]]

# print(max_contour_outter)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, ori.size)]

# axs = 3
# # ori_space = np.zeros((ori.size,axs))# space of points with the last one as color
# # index = 0
# # for y in range(ori.shape[0]):
# #     for x in range(ori.shape[1]):
# #         ori_space[index] = np.array([x,y,ori[y,x]])
# #         index += 1
#
# ori_space = np.array([[x,y,ori[y,x]] for x in range(ori.shape[1]) for y in range(ori.shape[0])])
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import kneighbors_graph
# from sklearn.cluster import DBSCAN
# # from sklearn.cluster import KMeans
# # from sklearn.cluster import AgglomerativeClustering
# from sklearn.mixture import GMM
#
# ori_space = StandardScaler().fit_transform(ori_space)
#
# result = np.zeros((ori.shape[0],ori.shape[1],3))
# db = DBSCAN(eps=0.3, min_samples=20)
# # db = KMeans(init='k-means++', n_clusters=10, n_init=10)
#
# # # connectivity matrix for structured Ward
# # connectivity = kneighbors_graph(ori_space, n_neighbors=10, include_self=False)
# # # make connectivity symmetric
# # connectivity = 0.5 * (connectivity + connectivity.T)
# #
# # gmm = GMM(n_components=3).fit(ori_space)
# # db = AgglomerativeClustering(n_clusters=3, linkage='ward',connectivity=connectivity)
# db = db.fit(ori_space)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# # labels = gmm.predict(ori_space)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Black removed and is used for noise instead.
# # print(labels)
# unique_labels = set(labels)
# # print(unique_labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
#
# ori_space[:,0] = cv.normalize(ori_space[:,0], None, alpha=0, beta=29, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1).reshape(-1)
# ori_space[:,1] = cv.normalize(ori_space[:,1], None, alpha=0, beta=49, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1).reshape(-1)
#
# ori_space[:,2] = cv.normalize(ori_space[:,2], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1).reshape(-1)
#
# import matplotlib.colors as mcolors
#
# ori_space = ori_space.astype(np.uint8)
# print(ori_space)
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     # xy = ori_space[class_member_mask]
#     xy = ori_space[class_member_mask & core_samples_mask]
#     # ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s=30)
#     for val in xy:
#         result[val[1], val[0]] = mcolors.to_rgba(col)[:3]
#     # xy = ori_space[class_member_mask]
#     xy = ori_space[class_member_mask & ~core_samples_mask]
#     # ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], 'o', c=tuple(col), s=15)
#     for val in xy:
#         result[val[1], val[0]] = mcolors.to_rgba(col)[:3]
#
# # ax.scatter(ori_space[:, 0], ori_space[:, 1], ori_space[:, 2], 'o', s=30)
#
# # plt.show()
# result = cv.resize(result, (300,500))
# cv.imshow('result', result)

from skimage import filters

source2 = ori.copy()
# source[label] = 0
# source[np.invert(label)] = 255
plt.imshow(source2); plt.show()
from skimage.segmentation import clear_border
c = max_contour_outter
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

size = extRight[0]-extLeft[0]
# label2 = label[extTop[1]:extTop[1]+size,extLeft[0]:extRight[0]]
ori = ori[extTop[1]:extTop[1]+size,extLeft[0]:extRight[0]]
ori = clear_border(ori)
from skimage.filters import try_all_threshold
fig, ax = try_all_threshold(ori, figsize=(10, 8), verbose=False)
# plt.show()
val = filters.threshold_otsu(ori)
mask = ori < 1.1*(val)
ori[mask] = 0
ori[np.invert(mask)] = 255
kernel = np.ones((3,3),np.uint8)
kernel_dilute = np.ones((5,5),np.uint8)
ori = cv.erode(ori,kernel,iterations = 5)
ori = cv.dilate(ori,kernel_dilute,iterations = 3)


# ori = cv.morphologyEx(ori, cv.MORPH_CLOSE, kernel)
# source[ori!=source] = 255
_, contours, _ = cv.findContours(ori.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# label = cv.cvtColor(label,cv.COLOR_GRAY2RGB)
# ori = cv.cvtColor(ori,cv.COLOR_GRAY2RGB)
max_contour = 0 # the index
for i, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    area_ref = cv.contourArea(contours[max_contour])
    # print(area, area_ref)
    if (area > area_ref): max_contour = i
max_contour = contours[max_contour]
# inter = label-ori
mask = np.zeros_like(ori)
label = np.zeros_like(mask)
cv.drawContours(mask, [max_contour_outter], -1, 2, -1)
cv.drawContours(mask, [max_contour], -1, 1, -1)
cv.drawContours(label, [max_contour], -1, 1, -1)
# cv.imshow('result', cv.normalize(distance_transform_edt(inter), None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F))
plt.imshow(mask); plt.show();
# sum = 0
# n = 0
from math import ceil
# for row in mask:
#     a = np.where(row==2)[0]
#     b = np.where(row==1)[0]
#     if a.size == 0 or b.size == 0: continue
#     a1 = a[0]#first of the array a
#     b1 = b[0]
#     if (a1<b1):
#         length = b1-a1
#         sum += length
#         n += 1
#
#     a2 = a[-1]#first of the array a
#     b2 = b[-1]
#     if (a2>b2):
#         length = a2-b2
#         sum += length
#         n += 1
# average = sum/n
#
# print(ceil(average))
# # cv.drawContours(mask, [max_contour_outter], -1, 2, -1)
# from skimage import segmentation
# # footprint = ndimage.generate_binary_structure(2, 1)
# # mask = segmentation.clear_border(mask).astype(np.uint8)
# res = ndimage.grey_erosion(mask, size=(ceil(average*1.5),ceil(average*1.5)), mode='constant')

threshold = 15
x = int(mask.shape[0]//1)-1
source = mask.copy()
source_parts = [source[i-x:i] if (i+x<=source.shape[0]) else source[i-x:source.shape[0]] for i in range(x,source.shape[0],x)]

res_total = np.zeros_like(source)

index = 0
# for i in source_parts:
#     sum = 0
#     n = 0
#
#     for row in i:
#         a = np.where(row==2)[0]
#         b = np.where(row==1)[0]
#         if a.size == 0 or b.size == 0: continue
#         a1 = a[0]#first of the array a
#         b1 = b[0]
#         if (a1<b1):
#             length = b1-a1
#             sum += length
#             n += 1
#
#         a2 = a[-1]#first of the array a
#         b2 = b[-1]
#         if (a2>b2):
#             length = a2-b2
#             sum += length
#             n += 1
#     if n==0: continue
#     average = ceil(sum/n)
#     i[i==2] = 1
#     res = ndimage.grey_erosion(i, size=(ceil(average*2),ceil(average*2)), mode='constant')
#     foo = index+x if index+x+x<=source.shape[0] else source.shape[0]
#     res_total[index:foo] = res
#     index += x
#     # print(res)
#     # res = ndimage.binary_erosion(source, structure=np.ones((average,average)))
#     # print(res)
#     print(ceil(average))
# print(res_total)


for i in source_parts:
    left = np.array([0,0])
    right = np.array([0,0])
    left_max = 0
    right_max = 0
    middle = np.array([0,0]) # first is sum, second is n

    for foo, row in enumerate(i):
        go_ = False
        b = np.where(row==1)[0]
        a = np.where(row==2)[0]
        c = np.where(row==0)[0]
        # if (b.size==0):
        #     i[foo] = np.zeros_like(row)
        if a.size == 0 or b.size == 0: continue
        pivot1 = b[0]
        pivot2 = b[-1]
        if c.size != 0:
            pivot = c[c<pivot1][-1] if c[c<pivot1].size != 0 else a[0]
            # print(pivot)
            a_left = a[a>pivot]
            # print(pivot2)
            pivot = c[c>pivot2][0] if c[c>pivot2].size != 0 else a[-1]
            a_right = a[a<pivot]
            a = np.concatenate([a_left,a_right])

        a1 = a[0]#first of the array a
        b1 = b[0]
        con_diff = b[-1]-b[0]
        # print(con_diff)
        if (a1<b1):
            length = b1-a1
            # print(foo,length)
            if (con_diff > length):
                left[0] += length
                left[1] += 1
                if (left_max<length):
                    left_max = length
                go_ = True

        a2 = a[-1]#first of the array a
        b2 = b[-1]
        if (a2>b2):
            length = a2-b2
            # print(foo,length)
            if (con_diff > length):
                right[0] += length
                right[1] += 1
                if (right_max<length):
                    right_max = length
                go_ = go_ and True
        # if (go_ == True):
        #     middle[0] += (b1+(b2-b1)//2)
        #     middle[1] += 1

    i[i==2] = 1
    alpha = 0.7
    # global_average = ceil((left[0]+right[0])//(left[1]+right[1]))
    # global_average =
    av_middle = ceil(middle[0]//middle[1]) if middle[1]!= 0 else 0#middle to seperate left and right
    print(av_middle)
    if (av_middle == 0):
        average = ceil(alpha*(left[0]+right[0])//(left[1]+right[1]) + (1-alpha)*max(left_max,right_max))
        # print(average)
        # average = ceil((alpha)*average+(1-alpha)*global_average)
        # print(average)
        res = ndimage.grey_erosion(i, size=(average*1.5,average*1.5), mode='constant', cval=0)
    else:
        i_left = i[...,0:av_middle]
        i_right = i[...,av_middle:i.shape[1]]
        # average_right = ceil((right[0])//(right[1]))
        average_left = ceil((1-alpha)*left_max+alpha*ceil((left[0])//(left[1])))
        average_right = ceil((1-alpha)*right_max+alpha*ceil((right[0])//(right[1])))
        # average_left = ceil((1-alpha)*average_left+(alpha)*global_average)
        # average_right = ceil((1-alpha)*average_right+(alpha)*global_average)

        print(average_left, average_right)
        # , mode='constant', cval=0
        res_left = ndimage.grey_erosion(i_left, size=(average_left,average_left), mode='constant', cval=0)
        res_right = ndimage.grey_erosion(i_right, size=(average_right,average_right), mode='constant', cval=0)
        res = np.zeros_like(i)
        res[...,0:av_middle] = res_left
        res[...,av_middle:i.shape[1]] = res_right
        # res = np.stack([res_left,res_right], axis=1).reshape(i.shape[0],-1)
        # print(res)
    foo = index+x if index+x+x<=source.shape[0] else source.shape[0]
    res_total[index:foo] = res
    # print(res_total)
    index += x




res = res_total
# plt.imshow(mask); plt.show();
# plt.imshow(res); plt.show();

cv.drawContours(mask, [max_contour_outter], -1, 3, -1)
# cv.drawContours(res2, [max_contour], -1, 1, -1)
kernel = np.ones((2,2),np.uint8)
mask = cv.dilate(mask, kernel, iterations=3)
res = cv.erode(res, kernel, iterations=5)
res2 = np.zeros_like(res)
cv.drawContours(res2, [max_contour], -1, 1, -1)
cv.drawContours(mask, [max_contour], -1, 2, -1)
# mask = cv.dilate(mask, kernel, iterations=4)
print(res.max())
res2 = ((1-res2) & res)
# res3 = res2.copy()
# kernel = np.ones((5,5),np.uint8)
# res3 = cv.dilate(res3, kernel, iterations=3)
#
# mask[res3>0] = 3
mask[res2>0] = 1
plt.imshow(mask);plt.show()
c = max_contour_outter
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

size = extRight[0]-extLeft[0]+10
mask2 = mask.copy()
mask = np.zeros_like(mask)
mask[extTop[1]:extTop[1]+size,extLeft[0]:extLeft[0]+size] = mask2[extTop[1]:extTop[1]+size,extLeft[0]:extLeft[0]+size]
# ori2 = ori[extTop[1]:extTop[1]+size,extLeft[0]:extRight[0]]

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
plt.imshow(mask);plt.show()
img = cv.cvtColor(source1.copy(),cv.COLOR_GRAY2RGB)
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==1)|(mask==3),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]

plt.imshow(img);plt.show()

# label = cv.cvtColor((1-label.copy())*255,cv.COLOR_GRAY2RGB)
# mask = cv.cvtColor(mask*255,cv.COLOR_GRAY2RGB)


# kernel = np.ones((3,3),np.uint8)
# mask = cv.erode(mask, kernel, iterations=5)
# kernel = np.ones((5,5),np.uint8)
# img = cv.dilate(img, kernel, iterations=3)
# res = cv.cvtColor(res,cv.COLOR_GRAY2RGB)
# res2 = cv.cvtColor((res2),cv.COLOR_GRAY2RGB)
source1 = source1*(1-res2[:,:])
# res2 = cv.cvtColor((res2),cv.COLOR_GRAY2RGB)

img = source1.copy()
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(max_contour_outter, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

plt.imshow(img),plt.colorbar(),plt.show()

# from skimage import morphology
# from skimage.morphology import watershed
# from skimage.feature import peak_local_max
# from scipy.ndimage.measurements import watershed_ift
# # # noise removal
# # img = label.astype(np.uint8)
# #
# # kernel = np.ones((3,3),np.uint8)
# # opening = cv.dilate(img, kernel)
# # # sure background area
# # sure_bg = cv.erode(img,kernel,iterations=3)
# # # Finding sure foreground area
# # dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# # ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # # Finding unknown region
# # sure_fg = np.uint8(sure_fg)
# # unknown = cv.subtract(sure_bg,sure_fg)
# #
# # # Marker labelling
# # ret, markers = cv.connectedComponents(sure_fg)
# # # Add one to all labels so that sure background is not 0, but 1
# # markers = markers+1
# # # Now, mark the region of unknown with zero
# # markers[unknown==255] = 0
# #
# # img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
# # markers = cv.watershed(img,markers)
# # img[markers == -1] = [255,0,0]
# #
# # cv.imshow('img',img)
#
# # label = label.astype(np.bool_)
# image=label
# # image = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# from scipy import ndimage
# distance = ndimage.distance_transform_edt(image)
# # distance = ndimage.maximum_filter(distance, size=10, mode='constant')
# ret, local_maxi = cv.threshold(distance,0.6*distance.max(),255,0)
# # local_maxi = peak_local_max(distance, min_distance=10, indices=False)
# # markers = morphology.label(local_maxi)
# labels_ws = watershed(ori, local_maxi)
# distance = cv.normalize(distance, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
# labels_ws = cv.normalize(labels_ws, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
# # markers = cv.normalize(markers, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
# # local_maxi = cv.normalize(local_maxi.astype(np.uint8), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
# # print(labels_ws)
# # cv.imshow('distance',distance)
# # cv.imshow('ori',ori)
# # cv.imshow('markers',markers)
# # cv.imshow('labels_ws', labels_ws)
# # cv.imshow('local_maxi', local_maxi)

# cv.imshow('test',local_maxi)

# cv.waitKey(0)
# sample[sample>0] = 1
# img = cv.imread("sample_2ds/0.png")
# sample = cv.resize(sample, (100, 50))
# # img = cv.resize(img, (100, 50))
# seg_boundaries = find_boundaries(sample, mode='inner')
#
# bin_img = img > 0
# binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)
#
# seg_boundaries = binary_with_borders.astype(np.uint8)
# seg_boundaries *= 255
# print(seg_boundaries)

# cv.imshow('result',construct_weights_and_mask(sample)[0])

cv.waitKey(0)

cv.destroyAllWindows()
