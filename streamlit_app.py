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

def final_fun_2(images):
     batch_images = np.stack(images, axis=0)    

     yhats = model.predict(batch_images)
     yhats = np.argmax(yhats, axis=1)
     
     return yhats

st.title("Real Estate Images Scene Classifier")
st.title("Using Deep Learning")
st.text("This classifier is used to classify real estate scene")
st.text("like bedroom, livingroom, bathroom, kitchen, frontyard and backyard")
#############################################################################################
with st.form('uploader'):
     # st.write(Enter your review below)
     image_file = st.file_uploader("Upload Your Real Estate Image Here", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
     # st.write("filename:", image_file.name)
     submitted = st.form_submit_button('Get image scene name..')


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
          
          yhats = final_fun_2(images)          

          predicted_class_names = ['This is a: '+str(class_names[i]) for i in yhats]

          st.image(images, caption=predicted_class_names)
     else:
          st.text('Please upload an image first then click on "Get images scene name" button!')
     
     
else:
     st.text('')

