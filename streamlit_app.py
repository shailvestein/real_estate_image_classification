import streamlit as st
import pickle
import keras
import rei_functions as rf
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

TARGET_SHAPE = (224,224)

#############################################################################################

# loading class names list
class_names = rf.get_class_names()

# loading trained model
model = rf.get_model_ready()
model.load_weights('./efficientNetB0_model.h5')

#############################################################################################
with st.form():
     # st.write(Enter your review below)
     image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
     # st.write("filename:", image_file.name)
     submitted = st.form_submit_button('Get image scene names..')


if not image_file is None:
     images = []
     for image in image_file:
          img = Image.open(image)
          img = rf.preprocess_image(img)
          img = img.resize(TARGET_SHAPE)
          images.append(img)

     #############################################################################################
if submitted:
     if len(images) > 0:
          st.text(len(images))
          # batch_images = np.array([np.stack(img) for img in images])
          batch_images = np.stack(images, axis=0)
          st.text(batch_images.shape)   
     else:
          st.text('Please upload an image!')

          
     yhats = model.predict(batch_images)
     yhats = np.argmax(yhats, axis=1)
     st.text(yhats)

     st.text('Please upload an real estate image!')
     #############################################################################################
     col = 3
     row = len(images)//col + 1
     rem = len(images) % col
     st.text(row)
     if rem > 0:
          row += 1

     # plt.subplots(

     st.text(row)
     st.text(rem)

