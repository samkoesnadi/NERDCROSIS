import numpy as np
from math import ceil
source = np.array([[0,0,0,0,0,0,0],[0,0,1,0,1,0,0],[0,1,1,2,2,0,0],[0,2,1,1,2,0,0],[0,1,0,2,2,1,0],[0,1,1,2,1,1,0],[0,0,0,0,0,0,0]])

x = source.shape[0]//2
source_parts = [source[i-x:i] if (i+x<=source.shape[0]) else source[i-x:source.shape[0]] for i in range(x,source.shape[0],x)]

print(source_parts)
from scipy import ndimage

res_total = np.zeros_like(source)

index = 0
for i in source_parts:
    left = np.array([0,0])
    right = np.array([0,0])
    middle = np.array([0,0]) # first is sum, second is n
    for row in i:
        a = np.where(row==1)[0]
        b = np.where(row==2)[0]
        if a.size == 0 or b.size == 0: continue
        a1 = a[0]#first of the array a
        b1 = b[0]
        if (a1<b1):
            length = b1-a1
            left[0] += length
            left[1] += 1

        a2 = a[-1]#first of the array a
        b2 = b[-1]
        if (a2>b2):
            length = a2-b2
            right[0] += length
            right[1] += 1

        if ((a1<b1) and (a2>b2)):
            middle[0] += b1+(b2-b1)
            middle[1] += 1


    av_middle = middle[0]//middle[1] if middle[1]!= 0 else 0#middle to seperate left and right
    if (av_middle == 0):
        average = ceil((left[0]+right[0])//(left[1]+right[1]))
        print(average)
        res = ndimage.grey_erosion(i, size=(average,average))
    else:
        i_left = i[...,0:av_middle]
        i_right = i[...,av_middle:i.shape[1]]
        average_left = ceil((left[0])//(left[1]))
        average_right = ceil((right[0])//(right[1]))
        print(average_left,average_right)
        # print(i_left, i_right, average_left, average_right)
        res_left = ndimage.grey_erosion(i_left, size=(average_left,average_left))
        res_right = ndimage.grey_erosion(i_right, size=(average_right,average_right))
        res = np.zeros_like(i)
        res[...,0:av_middle] = res_left
        res[...,av_middle:i.shape[1]] = res_right
        # res = np.stack([res_left,res_right], axis=1).reshape(i.shape[0],-1)
        # print(res)
    foo = index+x if index+x+x<=source.shape[0] else source.shape[0]
    res_total[index:foo] = res
    # print(res_total)
    index += x
    # print(res)
    # res = ndimage.binary_erosion(source, structure=np.ones((average,average)))
    # print(res)
    # print(ceil(average))
