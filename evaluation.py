import numpy as np
import itk
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from os import system as bash
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0
from os.path import join


testing_dir = "testing"
nifty_dir = "app"
prediction_dir = "prediction"


### 1. change the config directory
# instantiate
config = ConfigParser()

#all directories
config_file = join(nifty_dir,'extensions/configs/config1test.ini')
ori_dir = join(testing_dir,'volumes')
model_dir = join(nifty_dir,'models/1')
save_seg_dir = 'output_for_evaluation'

# parse existing file
config.read(config_file)

# # update existing value
config.set('image', 'path_to_search', ori_dir)
config.set('SYSTEM', 'model_dir', model_dir)
config.set('INFERENCE', 'save_seg_dir', save_seg_dir)
config.set('SEGMENTATION', 'num_classes', str(3))

# save to a file
with open(config_file, 'w') as configfile:
    config.write(configfile)


### 2. run main network for femur segmentation
#success = bash("net_segment inference -c "+config_file)

### 3. run necrosis segmentation and accuracy
from app.threed_necro_segment import main_necro_segment
f = open(join(prediction_dir,"log_accuracy.txt"),"w+")
f.seek(0)

f.write("Evaluation:\n\n")
for prediction, name in main_necro_segment(join(model_dir,save_seg_dir), ori_dir, join(prediction_dir,'necro_seg')):

    '''
    # prediction
    prediction = prediction.flatten()

    # just see necrosis
    np.where((prediction==3)|(prediction==4),prediction,0)
    prediction[prediction>0] = 1

    # get ori
    ori_itk = itk.imread(join(testing_dir,join('necro_seg',name)), 0)
    ori = itk.GetArrayFromImage(ori_itk)
    gt = ori.flatten()
    
    # jut see necrosis as one
    gt[gt>0] = 1
    
    f.write("- "+name+"::\n")
    cm = confusion_matrix(gt, prediction)
    print("Confusion matrix:\n",cm)
    print("Confusion matrix:\n",cm, file=f)


    acc = accuracy_score(gt, prediction)
    print("Accuracy = %s" % acc)
    f.write("Accuracy = %s\n" % acc)

    dsc = f1_score(gt, prediction)
    print("DSC = %s" % dsc)
    f.write("DSC = %s\n" % dsc)
    f.write("\n\n")
    '''

    ind = (prediction==3) | (prediction==4);
    prediction_bin = prediction*0;
    prediction_bin[ind] = 1;
    

    ori_itk = itk.imread(join(testing_dir,join('necro_seg',name)), 0)
    ori = itk.GetArrayFromImage(ori_itk)
    gt = ori > 0;

    prediction= prediction_bin.flatten()
    gt = gt.flatten()

    f.write("- "+name+"::\n")
    cm = confusion_matrix(gt, prediction)
    print("Confusion matrix:\n",cm)
    print("Confusion matrix:\n",cm, file=f)


    acc = accuracy_score(gt, prediction)
    print("Accuracy = %s" % acc)
    f.write("Accuracy = %s\n" % acc)

    dsc = f1_score(gt, prediction, average=None)
    print("DSC = %s" % dsc)
    f.write("DSC = %s\n" % dsc)
    f.write("\n\n")
