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

from keras.initializers import glorot_normal

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
    im1 = image.load_img(im_path, target_size=(96, 96),color_mode='grayscale')
    im1 = image.img_to_array(im1)
    im1 = np.expand_dims(im1, axis=0)
    im1 = im1/255.
#     im1 = preprocess_input(im1,mode='tf')
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

def getMiniBatch(batch_size=16, data=train):
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

def custom_error_function(y_true, y_pred):
    bool_finite = tf.is_finite(y_true)
    return K.mean(K.square(tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)), axis=-1)

def create_model(input_shape, output_len, optimizer, loss, metrics):
    alpha = 0.05
    dropout = 0.55
    
    filters = [32, 64, 128, 256] # [16, 32, 64, 128]
    kernel_size = [3, 2, 3, 2]
    pool_size = [3, 2, 3, 2]

    inp = Input(shape=(input_shape))
    x = Conv2D(filters=filters[0], kernel_size=kernel_size[0], padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(filters=filters[0], kernel_size=kernel_size[0], padding='same')(x)
    x = MaxPooling2D(pool_size=pool_size[0])(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=filters[1], kernel_size=kernel_size[1], padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(filters=filters[1], kernel_size=kernel_size[1], padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(filters=filters[1], kernel_size=kernel_size[1], padding='same')(x)
    x = MaxPooling2D(pool_size=pool_size[1])(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=filters[2], kernel_size=kernel_size[2], padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(filters=filters[2], kernel_size=kernel_size[2], padding='same')(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= Conv2D(filters=filters[2], kernel_size=kernel_size[2], padding='same')(x)
    x= MaxPooling2D(pool_size=pool_size[2])(x)
    x= BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= Dropout(dropout)(x)
    x= Conv2D(filters=filters[3], kernel_size=kernel_size[3], padding='same')(x)
    x= BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= Conv2D(filters=filters[3], kernel_size=kernel_size[3], padding='same')(x)
    x= BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= Conv2D(filters=filters[3], kernel_size=kernel_size[3], padding='same')(x)
    x= BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= Dropout(dropout)(x)
    x= Conv2D(filters=output_len, kernel_size=kernel_size[3], padding='same')(x)
    x= MaxPooling2D(pool_size=pool_size[3])(x)
    x= BatchNormalization()(x)
    x= LeakyReLU(alpha=alpha)(x)
    x= GlobalAveragePooling2D()(x)
    x= Dense(output_len, activation='tanh')(x)
    model = Model([inp],x)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.load_weights("/home/datumx/data_science_experiments/detect_kinship/github_repo/kin_detection_kaggle/train_version_files/weights/facial_keypoint_detection.h5")
    return model    

mod = create_model(input_shape=(96,96,1),
    output_len=30, 
    optimizer=RMSprop(lr=0.0008), # learning rate is changed during training
    loss=custom_error_function, # 'mean_squared_error', 
    metrics=['accuracy'])


        
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
    new_mod = Model(mod.input,mod.layers[-8].output)
    return new_mod

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

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
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




input_shape = (96,96,1)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

base_network = create_base_network(input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)


# extract features from detector
x_detector = processed_a
shape_detector = processed_a.shape

# extract features from extractor , same with detector for symmetry DxD model
shape_extractor = processed_b.shape
x_extractor = processed_b

# rehape to (minibatch_size, total_pixels, filter_size)
x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2] , shape_detector[-1]])(x_detector)
x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]])(x_extractor)

# outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])

# rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
# signed square-root 

x = keras.layers.Lambda(signed_sqrt)(x)
# L2 normalization
x = keras.layers.Lambda(L2_norm)(x)

### 
### attach FC-Layer
###

x = Dense(100, activation="relu")(x)
x = Dropout(0.01)(x)

out = Dense(units=1,kernel_regularizer=keras.regularizers.l2(1e-8),kernel_initializer='glorot_normal',activation="sigmoid")(x)



model = Model([input_a, input_b], out)

model.compile(loss="binary_crossentropy", metrics=['acc',auc], optimizer=Adam(0.0001))


# model.load_weights("/home/datumx/data_science_experiments/detect_kinship/logs/kin_relation_2019_05_28_21_53_51966472/siamese_kins_detection_13-val_loss_0.4720-val_acc_0.7680.h5")
# train


print(model.summary())
# train_batches =batch_generator(data_train,batch_size=8)
# valid_batches =batch_generator(data_valid,batch_size=8)

def Generator(batch_size, data ):
    while True:
        yield getMiniBatch(batch_size=batch_size, data=data)

train_gen = Generator(batch_size=16,data=train)
val_gen = Generator(batch_size=16,data=val)

model.fit_generator(train_gen,steps_per_epoch=1000,use_multiprocessing=True,
          epochs=100,validation_data=val_gen,validation_steps=100,callbacks=callbacks)
