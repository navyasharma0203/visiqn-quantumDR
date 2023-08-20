import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
def load_model():
    model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with the actual path
    return model

# Preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Process an image using the loaded model
def process_image(image_path):
    model = load_model()
    image_array = preprocess_image(image_path)
    
    # Make predictions using the model
    predictions = model.predict(image_array)
    
    # Extract relevant information from predictions
    confidence = predictions[0][0] * 100
    
    if confidence >= 80:
        grade = 'Moderate'
    elif confidence >= 50:
        grade = 'Mild'
    else:
        grade = 'Severe'
    
    if grade == 'Severe':
        recommendation = 'Consult a specialist immediately'
    elif grade == 'Moderate':
        recommendation = 'Consult a specialist for further evaluation'
    else:
        recommendation = 'Monitor regularly and consult if symptoms worsen'
    
    # Create a result dictionary
    result = {
        'grade': grade,
        'confidence': confidence,
        'recommendation': recommendation
        # You can add more information here, such as heatmap data
    }
    
    return result

