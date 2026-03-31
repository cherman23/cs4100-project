# cs4100-project
CS 4100 Final Project!

## Local Setup: Get Started:

This project uses python 3.12 because it is the latest version supported by tensorflow, which we need because 
our data format is in TFRecord format.

### Necessary installs
- pip install torchvision
- pip install numpy
- pip install tensorflow OR pip install tensorflow-macos 

### Populate data locally 
- Download the objects.tfrecord file from https://drive.google.com/file/d/1YKlQE8oCPcr7lGOFeSl_UrMgt6HroPiq/view and put it in the ./data folder
- Run setup.py file 
- This will populate the data/img folder with the training images and data/chords/labels.json with a map from image file name to chord number

