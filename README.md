# Hurricane Harvey Damage Classification

This project uses a Flask-based API containerized with Docker to classify satellite images from Hurricane Harvey as either "damage" or "no_damage". We tested three different types of models, and expanded on the Alternate-Lenet-5 model to complete our Flask-based API. 

## Included Files

1.**`project_3.ipynb`**
   This notebook contains Part 1 and 2 of our project. It contains code to load the data into Python data structures, investigate the datasets to determine basic attributes of the images, and ensure data is split for training, validation and testing. It also performs rescaling so that it can be used for training/evaluation of the neural networks you will build in Part 2. Additionally, it tests three different models:
   
   - A dense (i.e., fully connected) ANN
   - The Lenet-5 CNN architecture
   - Alternate-Lenet-5 CNN architecture, described in the paper listed in our citations

 As our Alternate-Lenet-5 CNN model receives the highest accuracy of 98.7716%, we continued using this model for Part 3. 

2. **`api.py`**

   This file defines a Flask application with two routes:
   - **`model_info`**: Returns metadata about the model using a GET request.
   - **`classify_building_image`**: Accepts an image file and returns a damage classification using a POST request.

3. **`Dockerfile`**

   Creates a Docker image, installs dependencies, and copies the application files.

4. **`docker-compose.yml`**

   Simplifies use of the Docker container.

5. **`dataset/`**

   Contains sample damaged and not damaged satellite images.

## Building Instructions

Before building, ensure **Docker** and **Docker Compose** are installed on your machine. You can check this by typing, 
```
docker --version
docker-compose --version
```

Then, clone this repo by writing the following in terminal,
```
git clone git@github.com:nicoleolv/coe379l-project3.git
```
Now, in order to build the Docker images and start the container, 
```
docker-compose up -d --build
```

Now, the server is up and running! We can test our requests using the following curl commands. The first curl command is our GET request.

```
curl http://localhost:5000/summary
```

This will return,
```
{
  "accuracy": 0.9877163,
  "description": "A modified version of the LeNet-5 architecture, designed for improved image classification with adjustments to layer configurations and parameters. This model is based on the dataset containing satellite images from Texas after Hurricane Harvey.",
  "name": "alt-lenet-5",
  "version": "v1"
}
```

Now, we can test our POST request. The command below is the format of our curl command.
```
curl -X POST -F "image=@dataset/Project3/example_directory/example_file.jpeg" http://localhost:5000/inference
```
An actual example of this is seen as,
```
curl -X POST -F "image=@dataset/Project3/no_damage/-95.061275_29.831535.jpeg" http://localhost:5000/inference
```
This outputs,
```
{
  "prediction": "no_damage"
}
```
Once we are done running the server, we can close it by typing the following in terminal
```
docker-compose down
```

## Citations

Alternate-Lenet-5 CNN architecture: https://arxiv.org/pdf/1807.01688.pdf
