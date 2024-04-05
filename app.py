# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from efficientnet.tfkeras import preprocess_input
# import numpy as np

# app = Flask(__name__)

# model = load_model('EfficientNet_ModelWeights.keras')

# def preprocess_and_predict(model, img_path, target_size=(224, 224)):
#     # Load and preprocess the image
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Make prediction
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction) 

#     # Return the predicted class
#     return predicted_class

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     # Save the uploaded file temporarily
#     file_path = 'temp_image.jpg'
#     file.save(file_path)

#     # Make prediction
#     predicted_class = preprocess_and_predict(model, file_path)

#     # Return the predicted class as a response
#     return render_template('index.html', prediction=predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from efficientnet.tfkeras import preprocess_input
import numpy as np

# Load your machine learning model
@st.cache(allow_output_mutation=True)
def load_model():
    return load_model('EfficientNet_ModelWeights.keras')

# Prediction function
def preprocess_and_predict(model, img_path, target_size=(224, 224)):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    if img is None:
        print("Error: Image not loaded.")
        return None
    
    # Converting image to array and preprocessing using EfficientNet's preprocessing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predicting the class label
    preds = model.predict(img_array)
    predicted_label = np.argmax(preds[0])

    reverse_expression_labels = {v: k for k, v in expression_labels.items()}

    # Converting the predicted label index to its corresponding expression label
    predicted_expression_label = reverse_expression_labels[predicted_label]
    
    return predicted_expression_label
def main():
    st.title('Expresso - Image Prediction')

    # Display the custom HTML content
    with open("index.html", "r", encoding="utf-8") as file:
        html_code = file.read()
        st.components.v1.html(html_code, width=700, height=800)

    # Load the model
    model = load_model()

    # Check if the file uploader is used
    if st.file_uploader is not None:
        uploaded_file = st.file_uploader("Upload your image", type=['jpg', 'png'])
        if uploaded_file is not None:
            # Make prediction when the "Predict" button is clicked
            if st.button('Predict'):
                # Save the uploaded file temporarily
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.read())
                # Make prediction
                predicted_class = preprocess_and_predict(model, "temp_image.jpg")
                # Display prediction result
                st.write(f'Predicted Class: {predicted_class}')

if __name__ == '__main__':
    main()

