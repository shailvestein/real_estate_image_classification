import streamlit as st
import pickle
import keras
import rei_functions as rf
import numpy as np
from PIL import Image
import tensorflow as tf

TARGET_SHAPE = (224,224)

#############################################################################################

# loading class names list
class_names = rf.get_class_names()

# loading trained model
model = rf.get_model_ready()
model.load_weights('./efficientNetB0_model.h5')

#############################################################################################

# st.write(Enter your review below)
image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
# st.write("filename:", image_file.name)

images = []
if image_file is not None:
     for image in image_file:
          img = Image.open(image)
          img = rf.preprocess_image(img)
          img = img.resize(TARGET_SHAPE)
          images.append(img)

#############################################################################################

st.text(len(images))
# batch_images = np.array([np.stack(img) for img in images])
batch_images = np.stack(images, axis=0)
st.text(batch_images.shape)   
# st.image(img, width=300)

yhats = model.predict(batch_images)
yhats = np.argmax(yhats, axis=1)
st.text(yhats)

#############################################################################################

num = len(images)//3
rem = len(images) % 3

st.text(num)
st.text(rem)

