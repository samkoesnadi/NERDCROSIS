# combine label (join femur segmentation and necrosis segmentation )

import itk 
import numpy as np
import cv2 as cv
from math import floor, ceil
from os import listdir, makedirs
from os.path import isfile, join, exists
from re import search

source_folder = "niftynet/data_raw"
dest_folder = "niftynet/data"

files = sorted(listdir(source_folder))

for file in files:
    if file[-13:]!="_Coronal.gipl" : continue 
    prefix = file[0:-13]
    fLabel_name = prefix+"_fLabel.gipl"
    nLabel_name = prefix+"_nLabel.gipl"
    print("Analyzing: "+fLabel_name + " "+nLabel_name)
    fLabel = itk.GetArrayFromImage(itk.imread(join(source_folder,fLabel_name)))
    nLabel = itk.GetArrayFromImage(itk.imread(join(source_folder,nLabel_name)))

    # here is the processing of labels 
    nLabel[nLabel>0] += 2

    # combine both together
    Label = fLabel
    Label[nLabel>0] = nLabel[nLabel>0]

    Label_itk = itk.GetImageFromArray(Label)

    print("Writing end Label in ",join(dest_folder, prefix+"_Label.gipl"))
    itk.imwrite(Label_itk, join(dest_folder, prefix+"_Label.gipl"))
