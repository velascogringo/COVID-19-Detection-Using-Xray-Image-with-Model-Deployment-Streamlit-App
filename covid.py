#import modules
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
from PIL import Image

# Load the trained Keras model
model = load_model('covid_detection_model.h5')

# Create a Streamlit app interface
st.title('COVID-19 DETECTION')
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
predict_button = st.button("Predict")
prediction_text = st.empty()  # Create an empty placeholder for the prediction text
probability_text = st.empty()  # Create an empty placeholder for the probability

if uploaded_image is not None:
    # Preprocess the uploaded image
    img = Image.open(uploaded_image)
    img = img.resize((256, 256))  # Resize to match your model's input size

    # Ensure the image has three color channels (RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = np.array(img) / 255.0  # Convert to NumPy array and normalize pixel values to the range [0, 1]
    img = img.reshape((1, 256, 256, 3))

    # Display the uploaded image
    st.image(img[0], caption='Uploaded X-ray Image', use_column_width=True)

if predict_button:
    # Make sure the model is loaded correctly
    if model:
        if uploaded_image is not None:
            # Make a prediction using the loaded model
            prediction = model.predict(img)

            # Extract the probability of being COVID-19 positive
            probability = prediction[0][0]

            # Display the prediction result and probability
            if probability > 0.5:
                prediction_text.write(f"Prediction: COVID-19 POSITIVE")
                probability_text.write(f"Probability: {probability:.2%}")
            else:
                prediction_text.write(f"Prediction: NORMAL CHEST XRAY")
                probability_text.write(f"Probability: {1 - probability:.2%}")
    else:
        st.write("Model loading failed. Please check the x-ray file.") 
