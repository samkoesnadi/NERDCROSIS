'''
Implementation of second beat of Hearbeat Network
'''

# from keras.models import Model
# from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, Activation, BatchNormalization
from keras import models
from keras import layers


from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def second_network(x):
    def layer_finishing(y):
        y = layers.BatchNormalization()(y)
        y = layers.PReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        if _strides==(1,1):
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, dilation_rate=2, padding='same')(y)
        else:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

    def resblock(y, nb_channels, _strides=(1, 1)):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """

        if _strides==(1,1):
            y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, dilation_rate=2)(y)
        else:
            y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides)(y)
        y = layer_finishing(y)

        collectr = y

        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = layer_finishing(y)

        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        y = layers.add([collectr, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # add zero padding
    first_pad = 1
    x = layers.ZeroPadding2D(padding=(first_pad, first_pad))(x)

    # conv1
    conv1 = layers.Conv2D(4, kernel_size=(4, 4), strides=(2, 2))(x)
    conv1 = layer_finishing(conv1)
    conv1_1 = layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1))(conv1)
    conv1_1 = layer_finishing(conv1_1)

    # conv2
    conv2 = layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(conv1)
    conv2 = layer_finishing(conv2)
    conv2_1 = layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(conv2)
    conv2_1 = layer_finishing(conv2_1)

    # conv3
    #x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    conv3 = resblock(conv2_1, 16)
    conv3_1 = resblock(conv3, 16)

    # conv4
    conv4 = resblock(conv3_1, 32)

    encoded = conv4

    decoded = layers.Conv2DTranspose(32, (3, 3), strides=1, dilation_rate=2, use_bias=False)(encoded)
    decoded = layer_finishing(decoded)
    #decoded = layers.concatenate([decoded, layers.ZeroPadding2D(((2,2),(2,2)))(conv4)])
    decoded = layers.UpSampling2D((2,2))(decoded)
    decoded = layers.Conv2DTranspose(32, (3, 3), strides=1, dilation_rate=2, use_bias=False)(decoded)
    decoded = layers.Dropout(0.5)(decoded)
    decoded = layer_finishing(decoded)
    decoded = layers.Conv2DTranspose(32, (3, 3), strides=1, dilation_rate=2, use_bias=False)(decoded)
    decoded = layer_finishing(decoded)
    decoded = layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=1, dilation_rate=2)(decoded)
    decoded = layers.concatenate([decoded, layers.ZeroPadding2D(((12,13+first_pad),(12,13+first_pad)))(conv3)])
    decoded = layer_finishing(decoded)
    decoded = layers.Conv2DTranspose(8, kernel_size=(3, 3), strides=1, dilation_rate=2)(decoded)
    decoded = layers.concatenate([decoded, layers.ZeroPadding2D(((11,12+first_pad),(11,12+first_pad)))(conv2)])
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.5)(decoded)
    decoded = layers.Conv2DTranspose(4, kernel_size=(4, 4), strides=1, dilation_rate=2)(decoded)
    decoded = layer_finishing(decoded)
    decoded = layers.concatenate([decoded, layers.ZeroPadding2D(((13,14+first_pad),(13,14+first_pad)))(conv1)])
    decoded = layers.Dropout(0.5)(decoded)
    decoded = layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=1)(decoded)
    decoded = layer_finishing(decoded)
    
    decoded = layers.Conv2DTranspose(2, kernel_size=(2, 2), strides=1)(decoded)
    decoded = layer_finishing(decoded)
    decoded = layers.Conv2D(1, kernel_size=(1, 1), strides=1)(decoded)
    decoded = layer_finishing(decoded)
    decoded = layers.Cropping2D(((1,0),(1,0)))(decoded)

    return decoded


def get_isnecrosis_model_def():  # 2 blocks 3x3 conv, one block res with 2 blocks 3 x3 conv and 'skip' connections to '+' node

    channels, height, weight = 1, 60, 60  # 1 channel because it's grayscale, dimensions of the input, one weight per pixel, 60 x 60 is size of npy array
    # Input
    shapeof_input = (height, weight, channels)

    # input and output
    img_input = layers.Input(shape=shapeof_input)
    net_output = second_network(img_input)

    model = models.Model(img_input, net_output, name='resnet')

    return model

ready_model = get_isnecrosis_model_def()
# read_model.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras import optimizers
adam = optimizers.Adam(decay=1e-6)
ready_model.compile(optimizer=adam, loss=jaccard_distance_loss, metrics=[dice_coef])

#ready_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# this is how to read the numpy array

#npy loader for npy arrays renamed 0-32

import matplotlib.pyplot as plt
import numpy as np

numOfArrays = 17
netdata = None
labels = None

for i in range(numOfArrays+1):
    for c in range(2):
        
        try:
            d = np.load('npy_middle/{}_{}.npy'.format(i, c))
        except:
            continue
        x = d.shape
        numOfSlices = x[0]
        for slices in range(numOfSlices):
            if ((d[slices][0][np.where(d[slices][0]>0)]).size!=0):
                #d[slices][0][np.where(d[slices][0]==0)] = np.mean(d[slices][0][np.where(d[slices][0]>0)])
                d[slices][0][np.where(d[slices][0]==0)] = d[slices][0].max() 
            if (d[slices][2].max()!=0):            
                d[slices][2] = d[slices][2].astype('float32')
                d[slices][2] /= d[slices][2].max()  # scale masks to [0, 1]

            if netdata is None:
                netdata = np.array([d[slices][0]])
                labels = np.array([d[slices][2]])
            else:
                netdata = np.concatenate((netdata,[d[slices][0]]), axis=0)
                labels = np.concatenate((labels,[d[slices][2]]), axis=0)


x_train = netdata[0:-10]
y_train = labels[0:-10]
x_test = netdata[-10:]
y_test = labels[-10:]

####
imgs_bit_train = x_train.astype('float32')
mean = np.mean(imgs_bit_train)
std = np.std(imgs_bit_train)

imgs_bit_train -= mean
imgs_bit_train /= std

imgs_bit_test = x_test.astype('float32')
imgs_bit_test -= mean
imgs_bit_test /= std

x_train = imgs_bit_train
x_test = imgs_bit_test

###

print(x_train.shape)
#y_train = y_train.astype('float32') / y_train.max()
#y_test = y_test.astype('float32') / y_train.max()
x_train = np.reshape(x_train, (len(x_train), 60, 60, 1))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), 60, 60, 1))
x_test = np.reshape(x_test, (len(x_test), 60, 60, 1))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), 60, 60, 1))


from keras.callbacks import TensorBoard
    
ready_model.fit(x_train, y_train,
                epochs=10,
                batch_size=6,
                shuffle=True,
                validation_split=0.1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# serialize weights to HDF5
ready_model.save_weights("model_second_net4.h5")
print("Saved model to disk")


#print(decoded_imgs.shape)

# load weights into new model
#ready_model.load_weights("model_second_net4.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
#ready_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

decoded_imgs = ready_model.predict(x_test)

n = 10	
for i in range(n):
    # evaluate the model
    #scores = ready_model.evaluate(x_test[i], y_test[i], verbose=0)
    #print(scores.shape)
    #print("%s: %.2f%%" % (ready_model.metrics_names[1], scores[1]*100))


    plt.imshow(y_test[i].reshape(60,60)), plt.show()
    plt.imshow(decoded_imgs[i].reshape(60,60)), plt.show()
