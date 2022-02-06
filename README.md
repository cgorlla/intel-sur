# Intel x HDSI UCSD System Usage Reporting Research

This repository contains the code for our research at UCSD on predicting PC user behavior in collaboration with Intel Corporation.

You can read about the development of the Input Libraries that collected the data used to predict in the paper below.

Cyril Gorlla, Jared Thach, Hiroki Hoshida. Development of Input Libraries With Intel XLSDK to Capture Data for App Start Prediction. 2022. ⟨[hal-03527679](https://hal.archives-ouvertes.fr/hal-03527679)⟩

## Repository Overview
- `config\`: contains configuration files for various scripts, such as data and output locations
- `notebooks\`: contains EDA with visualizations and other helpful Jupyter Notebooks to better understand the data
- `src\`: contains the main data loading, analysis, and model building scripts
- `main.py`: Python script to execute data parsing, data cleaning, training, and testing

## `run.py`
This Python file contains the necessary code to parse and clean data from the Input Libraries detailed in the above paper, as well as to build a first order Hidden Markov model. Development on other models (e.g. LSTM, RNN) is in progress.

### Building `run.py`
To run: `python run.py {data} {analysis} {model}`

To just build the model: `python run.py data model`

To test: `python run.py test` 

This will load in test data in `test\testdata` and build the HMM prediction model off of it. The predictions of the test model will be stored in `data\out\`.

## `src\model\model.py`
This file contains the first order Hidden Markov model class that will be used for predicting future foreground applications. After splitting the data and fitting the training set to a `first_order_HMM` instance using `fit`, the model keeps track of the prior and posterior probabilities of the training set's foreground applications. When inputting an observation, `X`, to `predict`, the function returns a list of foregrounds, (of size `n_foregrounds`, with default value of 1) with the highest conditional probability given `X`'s inputted foreground application and the trained model's posterior probabilities. `accuracy` returns the accuracy of the `y_test` on `y_pred` by taking each true foreground application in `y_test` and checking whether or not it appears in its respective list of foregrounds in `y_pred`.


## Docker
A dockerfile is included and will create a Docker environment that allows for the successful execution of all code in this repository.
