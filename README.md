[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
# Methods in AI Research
This repository houses all the code for the Methods in AI Research course at Utrecht University.\
The course is divided into three assignments, each of which is in a separate folder.

## Assignment 1a
The assignment consists of implementing a simple neural network from scratch and training it on a dialoge dataset.\
It contains the two rule based baseline models in file `baselines.py`.
The keywords for these baselines are located in `keywords.json`.

The machine learning models used are:
- A Feed-forward Neural Betwork in file `FNN.py`
- A Support Vector Machine in file `SVM.py`
- A Decision Tree Classifier in file `DTC.py`

Functions used by multiple files are located in `utils.py`.

### Usage
To get results from the models simply run the dedicated file.
These files should completely run without any additional input.

Normally you would't add data files to your github, but since this simplifies the process of running the code we have included the dataset in the `data` folder.

## Assignments 1b & 1c
This repository contains the code for the first assignment of the Methods in AI Research course at Utrecht University. The assignment consists of implementing a dialog system to help someone finding a restaurant in Cambridge.

To use the dialog system simply run the `main.py` file. This will start the dialog system and you can start asking questions.
The new features we implemented are the following:
- Levenshtein edit distance for preference extraction
- A user can now restart the questions by saying 'restart' after the suggestion.
- A user can now receive a small amount of recommendations instead of one.
- A user is now able to change previous answers if there is no suggestions available.
- A user can now receive formal and informal reactions.
