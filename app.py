from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from efficientnet.tfkeras import preprocess_input
import numpy as np

app = Flask(__name__)

# Load your machine learning model here
model = load_model('EfficientNet_ModelWeights.keras')

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
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming you're receiving an image file for prediction
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Save the uploaded file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Make prediction
    predicted_class = preprocess_and_predict(model, file_path)

    # Return the predicted class
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
