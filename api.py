from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Load image from request
    file = request.files['file']
    image = Image.open(file.stream)
    
    # Resize image to the expected input shape (75x100)
    image = image.resize((100, 75))  # Note: Image.resize expects (width, height)
    image_array = np.array(image)
    image_array = image_array.reshape((-1, 75, 100, 3))  # Add batch dimension

    # Normalize the image (if required by your model)
    image_array = image_array / 255.0

    # Make prediction
    predictions = model.predict(image_array)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
