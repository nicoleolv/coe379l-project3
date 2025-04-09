from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('./models/model_alt_cnn2.keras')

@app.route('/summary', methods=['GET'])
def model_info():
    return jsonify({
        "version": "v1",
        "name": "alt-lenet-5",
        "description": "A modified version of the LeNet-5 architecture, designed for improved image classification with adjustments to layer configurations and parameters. This model is based on the dataset containing satellite images from Texas after Hurricane Harvey.",
        "accuracy": 0.9735053
    })

def preprocess_input(im):
    """
    Converts user-provided image into an array that can be used with the model.
    This function could raise an exception.
    """
    im = im.resize((150, 150))
    
    img_array = np.array(im) / 255.0  # Normalize the image to [0, 1]
    
    # Add an extra dimension for batch size
    return img_array.reshape(1, 150, 150, 3)

@app.route('/inference', methods=['POST'])
def classify_building_image():
    # Get the image file from the request
    im = request.files.get('image')  # Expecting the image in the 'image' field
    
    if not im:
        return jsonify({"error": "The `image` field is required"}), 400
    
    try:
        # Open the image from the binary data
        image = Image.open(io.BytesIO(im.read()))
        
        # Preprocess the image into the appropriate format
        data = preprocess_input(image)
        
        # Perform inference
        prediction = model.predict(data)
        
        # Convert prediction to class label (0 = no_damage, 1 = damage)
        class_idx = np.argmax(prediction, axis=1)[0]
        label = "damage" if class_idx == 0 else "no_damage"
        
        return jsonify({"prediction": label})
    
    except Exception as e:
        return jsonify({"error": f"Could not process the `image` field; details: {e}"}), 400

# Start the development server (make sure to use the correct port)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
