from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import backend as K
import pandas as pd
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing import *
from keras.callbacks import *
from keras.optimizers import *
from tqdm import tqdm
import glob
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
# from keras.applications.nasnet import preprocess_input, decode_predictions
# from keras.applications.densenet import preprocess_input, decode_predictions
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import utils
from utils import LRN2D

import cv2
from random import choice, sample
import datetime
from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import threading
from utils import ResizeImage
from keras.initializers import glorot_normal

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def outer_product(x):
    """
    calculate outer-products of 2 tensors

        args 
            x
                list of 2 tensors
                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args 
            x
                a tensor
    """
    return keras.backend.l2_normalize(x, axis=axis)

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


print("start training.....................................")
print("start training.....................................")

if os.path.exists("logs"):
    pass
else:
    os.mkdir("logs")
log_dir = "/home/datumx/data_science_experiments/detect_kinship/logs/"+"kin_relation" + "_" + str(datetime.datetime.now()).replace(" ","-").replace(":","-").replace(".","").replace("-","_")
os.mkdir(log_dir)

ROOT = "/home/datumx/data_science_experiments/detect_kinship/"


def read_img(im_path):
    im1 = image.load_img(im_path, target_size=(160, 160))
    im1 = image.img_to_array(im1)
    im1 = np.expand_dims(im1, axis=0)
    im1 = preprocess_input(im1,mode='tf')
    return im1[0]

from collections import defaultdict
#keeps all photos path in a dictionary
allPhotos = defaultdict(list)
for family in glob.glob(ROOT+"train/*"):
    for mem in glob.glob(family+'/*'):
        for photo in glob.glob(mem+'/*'):
            allPhotos[mem].append(photo)

#list of all members with valid photo
ppl = list(allPhotos.keys())
len(ppl)

data = pd.read_csv(ROOT+'train_relationships.csv')
data.p1 = data.p1.apply( lambda x: ROOT+'train/'+x )
data.p2 = data.p2.apply( lambda x: ROOT+'train/'+x )
print(data.shape)
data.head()

data = data[ ( (data.p1.isin(ppl)) & (data.p2.isin(ppl)) ) ]
data = [ ( x[0], x[1]  ) for x in data.values ]
len(data)

train = [ x for x in data if 'F09' not in x[0]  ]
val = [ x for x in data if 'F09' in x[0]  ]
len(train), len(val)

def getImages(p1,p2):
    p1 = read_img(choice(allPhotos[p1]))
    p2 = read_img(choice(allPhotos[p2]))
    return p1,p2

def getMiniBatch(batch_size=2, data=train):
    p1 = []; p2 = []; Y = []
    batch = sample(data, batch_size//2)
    for x in batch:
        _p1, _p2 = getImages(*x)
        p1.append(_p1);p2.append(_p2);Y.append(1)
    while len(Y) < batch_size:
        _p1,_p2 = tuple(np.random.choice(ppl,size=2, replace=False))
        if (_p1,_p2) not in train+val and (_p2,_p1) not in train+val:
            _p1,_p2 = getImages(_p1,_p2)
            p1.append(_p1);p2.append(_p2);Y.append(0) 
    return [np.array(p1),np.array(p2)], np.array(Y)

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1),use_bias=use_bias)(input_tensor)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1),use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides, use_bias=use_bias)(input_tensor)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm()(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x
    
def root_resnet_fpn(input_image,architecture="resnet50",stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((1, 1))(input_image)

    x = KL.Conv2D(64, (3, 3), strides=(1, 1),  use_bias=False)(x)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.ZeroPadding2D((1, 1))(x)
    
    x = KL.Conv2D(64, (3, 3), strides=(1, 1), use_bias=False)(x)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.ZeroPadding2D((1, 1))(x)
    
    x = KL.Conv2D(64, (3, 3), strides=(1, 1), use_bias=False)(x)
    x = BatchNorm()(x, training=train_bn)
    x = KL.Activation('relu')(x)    
    
    C1 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding="valid")(x)
    
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

        
    out = [C1, C2, C3, C4, C5]
    
    L0 = ResizeImage(to_tuple(1),interpolation='bilinear')(out[0])
    L1 = ResizeImage(to_tuple(1),interpolation='bilinear')(out[1])
    L2 = ResizeImage(to_tuple(2),interpolation='bilinear')(out[2])
    L3 = ResizeImage(to_tuple(4),interpolation='bilinear')(out[3])
    L4 = ResizeImage(to_tuple(8),interpolation='bilinear')(out[4])
    
    
    upsampled_pyramid = [L0,L1,L2,L3,L4]
    
    x = Concatenate()(upsampled_pyramid)
    
    
    return x



        
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def create_base_network(input_shape):
    myInput = Input(shape=input_shape)
    out = root_resnet_fpn(myInput)
    
    return Model(inputs=[myInput], outputs=out)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


checkpoint_path = log_dir+ "/siamese_"+"kins_detection"+"_{epoch:02d}-val_loss_{val_loss:.4f}-val_acc_{val_acc:.4f}.h5"

callbacks = [
# keras.callbacks.EarlyStopping(monitor='val_loss',
#                            patience=8,
#                            verbose=1,
#                            min_delta=1e-4),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
            keras.callbacks.TensorBoard(log_dir=log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]



input_shape = (160,160,3)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

base_network = create_base_network(input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)


x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(processed_a), GlobalAvgPool2D()(processed_a)])
x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(processed_b), GlobalAvgPool2D()(processed_b)])

x3 = Subtract()([x1, x2])
x3 = Multiply()([x3, x3])

x = Multiply()([x1, x2])

x = Concatenate(axis=-1)([x, x3])
x  = BatchNormalization()(x)

x = Dense(100, activation="relu")(x)
x = Dropout(0.01)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model([input_a, input_b], out)

model.compile(loss="binary_crossentropy", metrics=['acc',auc], optimizer=SGD(0.01))

# # extract features from detector
# x_detector = processed_a
# shape_detector = processed_a.shape

# # extract features from extractor , same with detector for symmetry DxD model
# shape_extractor = processed_b.shape
# x_extractor = processed_b

# # rehape to (minibatch_size, total_pixels, filter_size)
# x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2] , shape_detector[-1]])(x_detector)
# x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]])(x_extractor)

# # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
# x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])

# # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
# x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
# # signed square-root 

# x = keras.layers.Lambda(signed_sqrt)(x)
# # L2 normalization
# x = keras.layers.Lambda(L2_norm)(x)

# ### 
# ### attach FC-Layer
# ###

# x = Dense(100, activation="relu")(x)
# x = Dropout(0.01)(x)

# out = Dense(units=1,kernel_regularizer=keras.regularizers.l2(1e-8),kernel_initializer='glorot_normal',activation="sigmoid")(x)



# model = Model([input_a, input_b], out)

# model.compile(loss="binary_crossentropy", metrics=['acc',auc], optimizer=Adam(0.01))


# model.load_weights("/home/datumx/data_science_experiments/detect_kinship/logs/kin_relation_2019_05_28_21_53_51966472/siamese_kins_detection_13-val_loss_0.4720-val_acc_0.7680.h5")
# train


print(model.summary())
# train_batches =batch_generator(data_train,batch_size=8)
# valid_batches =batch_generator(data_valid,batch_size=8)

def Generator(batch_size, data ):
    while True:
        yield getMiniBatch(batch_size=batch_size, data=data)

train_gen = Generator(batch_size=2,data=train)
val_gen = Generator(batch_size=2,data=val)

model.fit_generator(train_gen,steps_per_epoch=1600,use_multiprocessing=True,
          epochs=100,validation_data=val_gen,validation_steps=800,callbacks=callbacks)


