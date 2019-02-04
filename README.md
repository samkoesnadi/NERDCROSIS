large files (models) are in Google Drive/projects/nerdcrosis

Results filename format
{index of frame}_{index of iteration in that specific frame}
# NERDCROSIS

Transversal Data:
P0013


---------------------------------
Clustering technic to detect neucrosis
- Color and position space...
[x,y,z,color]
- no color info
[x,y,z]

- DBSCAN <- it works
- Ward <- meh
- Agglomerative Clustering
- Gaussian Mixture <- good one


///////////////// idea ///////////// 2019
1. change dataset to be more specific
2. train a machine learning to learn color histogram distribution or fft


//// Unused ////

W_0, SIGMA = 10, 5


def construct_weights_and_mask(imgs):
    weight_maps = []
    for img in imgs:
        seg_boundaries = find_boundaries(img, mode='inner')
        # print(seg_boundaries)
        weight_map = np.ones_like(seg_boundaries)*W_0
        cell_ids = [x for x in np.unique(img) if x > 0]
        print(np.any(seg_boundaries), len(cell_ids))
        if (np.any(seg_boundaries) and len(cell_ids)>1):
            print("inside")
            bin_img = img > 0
            # take segmentations, ignore boundaries
            binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

            foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
            background_weight = 1 - foreground_weight

            # build euclidean distances maps for each cell:
            distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

            for i, cell_id in enumerate(cell_ids):
                distances[..., i] = distance_transform_edt(img != cell_id)

            # we need to look at the two smallest distances
            distances.sort(axis=-1)

            if len(cell_ids) != 0:
                weight_map *= np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))

                weight_map[binary_with_borders] = foreground_weight
                weight_map[~binary_with_borders] += background_weight
        print(weight_map.shape)
        weight_maps.append([weight_map])
    return np.array(weight_maps)

def construct_weights_and_mask(img):
    seg_boundaries = find_boundaries(img, mode='inner')

    bin_img = img > 0
    # take segmentations, ignore boundaries
    binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

    foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
    background_weight = 1 - foreground_weight

    # build euclidean distances maps for each cell:
    cell_ids = [x for x in np.unique(img) if x > 0]
    distances = np.zeros((img.shape[0], img.shape[1], len(cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        distances[..., i] = distance_transform_edt(img != cell_id)

    # we need to look at the two smallest distances
    distances.sort(axis=-1)

    weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
    weight_map[binary_with_borders] = foreground_weight
    weight_map[~binary_with_borders] += background_weight

    return weight_map, binary_with_borders

plt.imshow(construct_weights_and_mask(cv.imread("sample_2ds/4_Label.png",0))[0]), plt.show()



######### this is 3d ROtation Transform (but still not fully functional)
from math import atan,pi
from skimage.transform import rotate
vx,vy,vz,x,y,z = cv.fitLine(label_ready, cv.DIST_L2, 0,0.01,0.01)

print(vx,vy,vz,x,y,z)
theta_x = atan(vz/vy)
theta_y = atan(vz/vx)
theta_z = atan(vy/vx)

R = cv.Rodrigues(np.array([-(theta_x+pi/2), -(theta_y+pi/2), -(theta_z+pi/2)]))[0]

print(R)
from skimage.transform import AffineTransform, SimilarityTransform, warp, EuclideanTransform
shift_y, shift_x = np.array(label.shape[:2]) / 2.
tf_rotate = SimilarityTransform(rotation=theta)
tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

new_label = warp(label, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
plt.imshow(new_label), plt.show()
# for a in range(10):
#     plt.imshow(hasil[a]), plt.show()
