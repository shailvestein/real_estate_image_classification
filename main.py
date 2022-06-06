import streamlit as st
import pickle
import keras

#############################################################################################
#############################################################################################

# Loading tfidf trained vectorizer to convert taken review in numbers

with open(vect.pkl, 'rb') as v
    vect = pickle.load(v)

# Defining and loading model weights for prediction of taken review

input_layer = keras.Input(shape=(2662,))
layer_1 = keras.layers.Dense(1000, activation='relu')(input_layer)
layer_2 = keras.layers.Dense(500, activation='relu')(layer_1)
output_layer = keras.layers.Dense(1, activation='sigmoid')(layer_2)
model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='IMBD')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights(4th_model_0_77_accuracy.h5)

##############################################################################################
##############################################################################################

# Below we are defining a streamlit webpage which will take user input and predict polarity of taken review

st.title(Movie review polarity prediction)
st.write(Here we will predict polarity of your review for a movie whether or not it is positive.)



with st.form(my_form)
    # st.write(Enter your review below)
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
         bytes_data = uploaded_file.read()
         st.write("filename:", uploaded_file.name)
         st.write(bytes_data)

    # Every form must have a submit button.
    submitted = st.form_submit_button(Predict)


if submitted
    if review == ''
        st.text(fPlease enter your review and press predict)
    else
        input_review = vect.transform([review])
        input_review = input_review.toarray()

        predicted_score = model.predict(input_review)

        if predicted_score  0.5
            st.text(fYou've given negative review)
        else
            st.text(fYou've given positive review)