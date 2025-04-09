# Hurricane Harvey Damage Classification

This project uses a Flask-based API containerized with Docker to classify satellite images from Hurricane Harvey as either "damage" or "no_damage". We tested three different types of models, and expanded on the Alternate-Lenet-5 model to complete our Flask-based API. 

## Included Files

1. **`api.py`**

   This file defines a Flask application with two routes:
   - **`model_info`**: Returns metadata about the model (GET request).
   - **`classify_image`**: Accepts an image file and returns a damage classification (POST request).

2. **`Dockerfile`**

   A recipe for creating a Docker image, installing dependencies (TensorFlow, Flask, Pillow) and copying the application files.

3. **`docker-compose.yml`**

   Simplifies management of the Docker container, defining the service and mapping port 5000.

4. **`models/model_alt_cnn2.keras`**

   The pre-trained Keras model file for image classification.

5. **`dataset/`**

   Directory containing sample satellite images (e.g., `dataset/Project3/no_damage/-95.061275_29.831535.jpeg`).

## Building Instructions

Ensure Docker and Docker Compose are installed. To build the Docker image:

```bash
docker-compose build
