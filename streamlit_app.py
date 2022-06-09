import streamlit as st
import numpy as np
import tensorflow as tf
import pickle as pkl

from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

#### Constants
H, W = 224, 224
INPUT_SHAPE = (H,W,3)
TARGET_SHAPE = (H,W)
NUM_CLASSES = 6

# function to get class names
@st.cache
def get_class_names():
    with open('./class_list.pkl', 'rb') as f:
        class_names = pkl.load(f)
    return class_names
# loading class names list
class_names = get_class_names()

# getting model ready
@st.cache
def get_model_ready():
    # loading pre-trained efficientNetB0
    efn = EfficientNetB0(include_top=False, 
                           weights='imagenet', 
                           input_shape=INPUT_SHAPE)

    # making each layers not to train
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




# title for the webpage
st.title("Real Estate Images Scene Classifier Using Deep Learning")
# text to describe about web app
st.text("Here you can classify real estate scene like bedroom,")
st.text("livingroom, bathroom, kitchen, frontyard and backyard.")
st.text("You just need to upload a new real estate image here")
st.text("It performs some mathematical magic on images and tells ")
st.text("which scene type it is.")

# creating form to upload image 
with st.form('uploader'):
     # file uploader
     image_file = st.file_uploader("Upload Your Real Estate Image Here", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
     # submit button
     submitted = st.form_submit_button('Get image scene name..')

# loading trained model
model = get_model_ready()

# loading trained weights
model.load_weights('./efficientNetB0_model.h5')

# function to predict class labels on images
def final_fun_2(images):
     # stacking images to form (batch_size, height, width, channel) format
     batch_images = np.stack(images, axis=0)    
     # classifiying images
     yhats = model.predict(batch_images)
     st.text(yhats)
     yhats = np.argmax(yhats, axis=1)
     t.text(yhats)
     # returning class names index
     return yhats
    
    
# appending images into list if there are more than 1 images uploaded
# if image_file is not none
if not image_file is None:
     # this list contains all uploaded images
     images = []
     # for each image from uploaded images
     for image in image_file:
          # reading image file
          img = Image.open(image)
          # applying preprocessing function
          img = preprocess_input(img)
          # resizing images
          img = img.resize(TARGET_SHAPE)
          # appending to the list
          images.append(img)

        


    
# if get images scene name button clicked
if submitted:
     # check if image is uploaded
     if len(images) > 0:
          # if image is uploaded pass this to final_fun_2() for classifying scene type
          yhats = final_fun_2(images)          
          # list to pass st.image() to print predicted class label name
          predicted_class_names = ['This is a: '+str(class_names[i]) for i in yhats]
          # printing image and respected predicted class label
          st.image(images, caption=predicted_class_names)
     else:
          # if get image scene name is clicked but no images are uploaded print this messege
          st.text('Please upload an image first then click on "Get images scene name" button!')
     
     
else:
     # print space character
     st.text('')

