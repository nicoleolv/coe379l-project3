from flask import Flask, request, jsonify
import tensorflow as tf 
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('./models/model_alt_cnn2.keras')

@app.route('/summary', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "alt-lenet-5",
      "description": "A modified version of the LeNet-5 architecture, designed for improved image classification with adjustments to layer configurations and parameters. This model is based on the dataset containing satellite images from Texas after Hurricane Harvey.",
      "accuracy": 0.9735053
   }


def preprocess_input(im):
   """
   Converts user-provided input into an array that can be used with the model.
   This function could raise an exception.
   """
   # convert to a numpy array
   d = np.array(im)
   # then add an extra dimension
   return d.reshape(1, 150, 150)

@app.route('/inference', methods=['POST'])
def classify_building_image():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
        # Preprocess the image
        data = preprocess_input(image_file)
        # Perform inference
        prediction = model.predict(data)
        # Convert prediction to class label (0 = no_damage, 1 = damage)
        class_idx = np.argmax(prediction, axis=1)[0]
        label = "damage" if class_idx == 1 else "no_damage"
        return {"prediction": label}
       
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": model.predict(data).tolist()}




# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=5001)

    