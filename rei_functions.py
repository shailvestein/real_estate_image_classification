
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import Conv2D, Input, Dense, GlobalAveragePooling2D, MaxPool2D, Dropout

from tensorflow.keras.models import Model

#### Constants
H, W = 224, 224
INPUT_SHAPE = (H,W,3)
TARGET_SHAPE = (H,W)
NUM_CLASSES = 6
  
  
with open('./class_list.pkl', 'rb') as f:
    class_names = pkl.load(f)

 
#############################################################################
#############################################################################


def get_model_ready():
    efn = EfficientNetB0(include_top=False, 
                           weights='imagenet', 
                           input_shape=INPUT_SHAPE)

    for layers in efn.layers:
        layers.trainable=False

    # input layer
    inputs = Input(shape=INPUT_SHAPE, name='input_shape')
    # passing input layer to resnet 50
    x = efn(inputs)
    # Global pulling layer
    x = GlobalAveragePooling2D(name='global_pooling')(x)
    # Dense layer
    x = Dense(1024, activation='relu', name='dense_1')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_1')(x)
    # Dense layer
    x = Dense(64, activation='relu', name='dense_2')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_2')(x)
    # output layer
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # intialising model
    efficientNet_model = Model(inputs=inputs, outputs=outputs, name='efficientNet_based_model')

    return efficientNet_model




#############################################################################
#############################################################################


