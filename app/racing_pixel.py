import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import grey_erosion
from math import ceil

def racing_pixel(source, n=1, alpha=0.7, beta=0.5, seperate_left_right=False, mode='constant'):
    '''
    Input
    source = original mask
    n = how much y-sections
    alpha = (1-percentage) of maxima of each section
    beta = (1-percentage) of global_average
    seperate_left_right = are we going to seperate_left_right
    mode = mode of erosion
    '''
    x = int(source.shape[0]//n)-1
    source_parts = [source[i-x:i] if (i+x<=source.shape[0]) else source[i-x:source.shape[0]] for i in range(x,source.shape[0],x)]

    res_total = np.zeros_like(source)

    index = 0
    lefts = []
    rights = []
    left_maxs = []
    right_maxs = []
    middles = []
    global_average = 0
    for i in source_parts:
        left = np.array([0,0])
        right = np.array([0,0])
        left_max = 0
        right_max = 0
        middle = np.array([0,0]) # first is sum, second is n

        for foo, row in enumerate(i):
            go_ = False
            b = np.where(row==2)[0]
            a = np.where(row==1)[0]
            c = np.where(row==0)[0]
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

            if (a1<b1):
                length = b1-a1
                if (con_diff > length):
                    left[0] += length
                    left[1] += 1
                    if (left_max<length):
                        left_max = length
                    go_ = True

            a2 = a[-1] #first of the array a
            b2 = b[-1]
            if (a2>b2):
                length = a2-b2
                if (con_diff > length):
                    right[0] += length
                    right[1] += 1
                    if (right_max<length):
                        right_max = length
                    go_ = go_ and True
            if (go_ == True and seperate_left_right == True):
                middle[0] += (b1+(b2-b1)//2)
                middle[1] += 1
        lefts.append(left)
        rights.append(right)
        left_maxs.append(left_max)
        right_maxs.append(right_max)
        middles.append(middle)

        global_average = (global_average+ceil((left[0]+right[0])//(left[1]+right[1]))) / 2 if global_average!=0 else ceil((left[0]+right[0])//(left[1]+right[1]))
    for i, part in enumerate(source_parts):
        left = lefts[i]
        right = rights[i]
        middle = middles[i]
        left_max = left_maxs[i]
        right_max = right_maxs[i]

        part[part>1] = 1 # important
        av_middle = ceil(middle[0]//middle[1]) if middle[1]!= 0 else 0#middle to seperate left and right
        if (av_middle == 0):
            # print(left_max,right_max)
            average = ceil(alpha*(left[0]+right[0])//(left[1]+right[1]) + (1-alpha)*max(left_max,right_max))
            average = ceil((beta)*average+(1-beta)*global_average)

            # res = grey_erosion(part, size=(average,average), mode=mode, cval=0)
            res = binary_erosion(part, iterations=average)
        else:
            i_left = part[...,0:av_middle]
            i_right = part[...,av_middle:part.shape[1]]

            average_left = ceil((1-alpha)*left_max+alpha*ceil((left[0])//(left[1])))
            average_right = ceil((1-alpha)*right_max+alpha*ceil((right[0])//(right[1])))
            average_left = ceil((1-beta)*average_left+(beta)*global_average)
            average_right = ceil((1-beta)*average_right+(beta)*global_average)

            res_left = grey_erosion(i_left, size=(average_left,average_left), mode=mode, cval=0)
            res_right = grey_erosion(i_right, size=(average_right,average_right), mode=mode, cval=0)
            # res_left = binary_erosion(i_left, iterations=average_left)
            # res_right = binary_erosion(i_right, iterations=average_right)
            res = np.zeros_like(part)
            res[...,0:av_middle] = res_left
            res[...,av_middle:part.shape[1]] = res_right
        foo = index+x if index+x+x<=source.shape[0] else source.shape[0]
        res_total[index:foo] = res
        index += x

    return res_total
