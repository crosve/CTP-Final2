import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# Load the model (assuming you have the model saved as nn.h5)
model = tf.keras.models.load_model("nn.h5")

# Compile the model (necessary to compute metrics)
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Image processing function for Streamlit using TensorFlow
def process_image_for_inference(image, img_size=(32, 32)):
    img = tf.keras.preprocessing.image.load_img(image, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array.reshape(1, 32,32, 3)


# Streamlit UI for fake image detection
st.title("Fake Image Detection")

# File uploader widget for users to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_processed = process_image_for_inference(uploaded_image, img_size=(32, 32))
    
    # Get the model prediction
    prediction = model.predict(img_processed)
    
    
    
    # Map to human-readable format
    label = "Real" if prediction > 0.5 else "Fake"
    
    # Display results
    st.write(f"Prediction: {label}")
else:
    st.write("Please upload an image to get started.")
   
