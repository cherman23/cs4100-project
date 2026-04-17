# cs4100-project
CS 4100 Final Project!

## Local Setup: Get Started:

This project uses python 3.12 because it is the latest version supported by tensorflow, which we need because 
our data format is in TFRecord format.

### Necessary installs
- Install the requirements.txt

### Populate data locally 
- Download the objects.tfrecord file from https://drive.google.com/file/d/1YKlQE8oCPcr7lGOFeSl_UrMgt6HroPiq/view and put it in the ./data folder
- Run `python processing/process_data.py` to fetch the data for the data
- This will populate the data/img folder with the training images and data/chords/labels.json with a map from image file name to chord number

### Running the CNN
- Run python models/cnn.py
- The terminal will output the epoch results and some other logs
- The result of model will be saved to a folder with the name of format model_ACCURACY_EPOCHS
  - In this folder will be the model and the data vis for the model and training results

### Running the Landmarking NN
- First populate the data
- Run `python models/nn.py` to run the neural network
- The accuracy of the network will be printed in the console

