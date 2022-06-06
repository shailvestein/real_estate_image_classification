import streamlit as st
import pickle
import keras
import rei_functions as rei

#############################################################################################
#############################################################################################

# trained model
model = rf.get_model_ready()
model.load_weights(efficientNetB0_model.h5)

##############################################################################################
##############################################################################################

# Below we are defining a streamlit webpage which will take user input and predict polarity of taken review



with st.form(my_form):
    # st.write(Enter your review below)
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
         bytes_data = uploaded_file.read()
         st.write("filename:", uploaded_file.name)
         st.write(bytes_data)
