import streamlit as st
import pickle
import keras
import rei_functions as rf
import numpy as np
from PIL import Image
import tensorflow as tf

TARGET_SHAPE = (224,224)
#############################################################################################
#############################################################################################
# loading class names list
class_names = rf.get_class_names()

# loading trained model
model = rf.get_model_ready()
model.load_weights('./efficientNetB0_model.h5')

##############################################################################################
##############################################################################################

# Below we are defining a streamlit webpage which will take user input and predict polarity of taken review




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


st.text(len(images))
batch_images = np.array([img for img in images])
st.text(batch_images.shape)   
# st.image(img, width=300)

# img = tf.expand_dims(img, axis=0)
# shape = img.shape
# st.text(shape)
# yhat = model.predict(img)

# predicted_class_names = class_names[np.argmax(yhat)]
# st.text(predicted_class_names)

# original_image = Image.open(image_file)
# original_image = np.array(original_image)

# img = Image.open(original_image)
# img = np.array(img)
# st.image([img])
     
# length = 12
# st.text(f"{length} images uploaded!")
