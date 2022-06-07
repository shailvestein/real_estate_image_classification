import streamlit as st
import pickle
import keras
import rei_functions as rf
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# shape of input image to resize
TARGET_SHAPE = (224,224)

# loading class names list
class_names = rf.get_class_names()

# loading trained model
model = rf.get_model_ready()
model.load_weights('./efficientNetB0_model.h5')

# function to predict class labels on images
def final_fun_2(images):
     # stacking images to form (batch_size, height, width, channel) format
     batch_images = np.stack(images, axis=0)    
     # classifiying images
     yhats = model.predict(batch_images)
     yhats = np.argmax(yhats, axis=1)
     # returning class names index
     return yhats

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
          img = rf.preprocess_image(img)
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

