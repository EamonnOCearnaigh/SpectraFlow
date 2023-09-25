# SpectraFlow


## Requirements

tensorflow>=2.13.0

numpy>=1.24.3

scipy>=1.11.1

pandas>=2.0.3

matplotlib>=3.7.2


## Instructions

Installing as a python package from the command line:

pip install spectraflow-1.0.0.tar.gzc

The setup.py file that created this file is included in this project directory.

I have provided a script, demonstration.py, that automatically imports the package scripts (which import the rest of the code),
and runs the three main scripts of the package:

train_model - For training new models

test_model - For testing these models against a series of metrics

predict_spectra - The use of a given model for predicting peptide spectra, producing an MGF file


## Structure of Source Code

spectraflow - main directory (and package)

predict_spectra - script for using a trained model for prediction

train_model - script for training a tensorflow model

test_model - script for evaluating a trained tensorflow model

data - contains all of the data for all three scripts

models - contains trained model weights

data_io - input and output python files

processing - general processing files

resources - files containing constants and related data (modifications, masses)

rnn - contains the BiLSTM tensorflow model code, as well as configurations and settings

