import streamlit as st
import pickle
import keras
import rei_functions as rf
import numpy as np
from PIL import Image

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
image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

# st.write("filename:", image_file.name)

if image_file is not None:
     img = Image.open(image_file)
     st.image(img)

# original_image = Image.open(image_file)
# original_image = np.array(original_image)

# img = Image.open(original_image)
# img = np.array(img)
# st.image([img])
     
# length = 12
# st.text(f"{length} images uploaded!")
