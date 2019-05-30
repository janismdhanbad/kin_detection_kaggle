import pandas as pd
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing import *
from keras.callbacks import *
from keras.optimizers import *
from tqdm import tqdm

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import cv2

import datetime

if os.path.exists("logs"):
    pass
else:
    os.mkdir("logs")
log_dir = "/home/datumx/data_science_experiments/detect_kinship/logs/"+"kin_relation" + "_" + str(datetime.datetime.now()).replace(" ","-").replace(":","-").replace(".","").replace("-","_")
os.mkdir(log_dir)

ROOT = "/home/datumx/data_science_experiments/detect_kinship/"

# data = pd.read_csv("train_data_images.csv")

# similar = data[data["similarity"] == 1].index.values
# dsimilar = data[data["similarity"] != 1].index.values

# sim = np.random.choice(similar,200000,replace=False)
# dsim = np.random.choice(dsimilar,200000,replace=False)

# all_data_sample = sim.tolist() + dsim.tolist()

# data = data.iloc[all_data_sample,:]

# data.index = range(data.shape[0])

# train,valid = train_test_split(data.index.values,test_size = 2000,stratify = data["similarity"].values)



# data_train = data.iloc[train,:]
# data_valid = data.iloc[valid,:]

# data_train.index = range(data_train.shape[0])
# data_valid.index = range(data_valid.shape[0])

# data_train.to_csv("tr_data.csv",index=False)
# data_valid.to_csv("val_data.csv",index=False)

data_train = pd.read_csv("/home/datumx/data_science_experiments/detect_kinship/tr_data.csv")
data_valid = pd.read_csv("/home/datumx/data_science_experiments/detect_kinship/val_data.csv")


def batch_generator(names,batch_size):
    total_range = np.arange(len(names)).tolist()
    if len(names)%batch_size != 0:
        to_add = batch_size - ((len(names))%batch_size)
        total_range = total_range + np.random.choice(total_range,to_add,replace=False).tolist()
    total_range = np.array(total_range)
    np.random.shuffle(total_range)
    temp_df = pd.DataFrame(total_range)
    
    temp_df.columns = ["batch"]
    temp_df["batch_ind"] = temp_df.index 
    temp_df["batch_ind"] = (temp_df["batch_ind"].astype('int')) //batch_size

    return temp_df.groupby('batch_ind')['batch'].apply(list).values




def generator(names,batches,batch_size):
    batch_num = 0
    while True:
        
        if batch_num >= len(names)/batch_size:
            batch_num = 0
        imgs1 = []
        imgs2 = []
        labels = []
#         print(batch_num)
        batch = batches[batch_num]
        for i in range(batch_size):
#             print (train_names[batch[i]][0], train_names[batch[i]][1],train_names[batch[i]][2] )
            img_name1 = names[batch[i]][0]
            img_name2 = names[batch[i]][1]
            label = names[batch[i]][2]
            im1 = image.load_img(ROOT+img_name1, target_size=(224, 224))
            im1 = image.img_to_array(im1)
            im1 = np.expand_dims(im1, axis=0)
            im1 = preprocess_input(im1,mode='tf')
            
            im2 = image.load_img(ROOT+img_name2, target_size=(224, 224))
            im2 = image.img_to_array(im2)
            im2 = np.expand_dims(im2, axis=0)
            im2 = preprocess_input(im2,mode='tf')

            imgs1.append(im1[0])
            imgs2.append(im2[0])
            labels.append(label)
        batch_num += 1    
        yield [np.array(imgs1),np.array(imgs2)], np.array(labels)    
        
        
        
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
    out = keras.applications.ResNet50(include_top=False,input_shape=(224,224,3))
    return Model(out.input,out.output)

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


input_shape = (224,224,3)

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

x = Dense(100, activation="relu")(x)
x = Dropout(0.01)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model([input_a, input_b], out)

model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.001))


model.load_weights("/home/datumx/data_science_experiments/detect_kinship/logs/kin_relation_2019_05_28_21_53_51966472/siamese_kins_detection_13-val_loss_0.4720-val_acc_0.7680.h5")
# train


print(model.summary())
train_batches =batch_generator(data_train,batch_size=8)
valid_batches =batch_generator(data_valid,batch_size=8)

model.fit_generator(generator(data_train.values,train_batches,batch_size=8),steps_per_epoch=24000,
          epochs=30,validation_data=generator(data_valid.values,valid_batches,batch_size=8),validation_steps=250,callbacks=callbacks)