# INTELli*next*: A Fully Integrated LSTM and HMM-Based Solution for Next-App Prediction With Intel SUR SDK Data Collection
# Intel DCA x HDSI UCSD System Usage Reporting Research

As the power of modern computing devices increases, so too do user expectations for them. Despite advancements in technology, computer users are often faced with the dreaded spinning icon waiting for an application to load. Building upon our previous work developing data collectors with the Intel System Usage Reporting (SUR) SDK, we introduce INTELli*next*, a comprehensive solution for next-app prediction for application preload to improve perceived system fluidity. We develop a Hidden Markov Model (HMM) for prediction of the k most likely next apps, achieving an accuracy of 70% when k = 3. We then implement a long short-term memory (LSTM) model to predict the total duration that applications will be used. After hyperparameter optimization leading to an optimal lookback value of 5 previous applications, we are able to predict the usage time of a given application with a mean absolute error of ~45 seconds. Our work constitutes a promising comprehensive application preload solution with data collection based on the Intel SUR SDK and prediction with machine learning.


This repository contains the code for our research at UCSD on predicting PC user behavior in collaboration with Intel Corporation.

You can read about the development of the Input Libraries that collected the data used to predict in the paper below.

Cyril Gorlla, Jared Thach, Hiroki Hoshida. Development of Input Libraries With Intel XLSDK to Capture Data for App Start Prediction. 2022. ⟨[hal-03527679](https://hal.archives-ouvertes.fr/hal-03527679)⟩

## Repository Overview
- `config\`: contains configuration files for various scripts, such as data and output locations
- `notebooks\`: contains EDA with visualizations and other helpful Jupyter Notebooks to better understand the data
- `src\`: contains the main data loading, analysis, and model building scripts
- `main.py`: Python script to execute data parsing, data cleaning, training, and testing

## `run.py`
This Python file contains the necessary code to parse and clean data from the Input Libraries detailed in the above paper, as well as to build the models in the project. These include:
- First Order Hidden Markov Model for Next-App Prediction
- LSTM for Next-App Prediction
- LSTM for App Duration Prediction

### Building `run.py`
To run: `python run.py {data} {analysis} {model}`

To just build the model: `python run.py data model`

To test: `python run.py test` 

This will load in test data in `test\testdata` and build the HMM and LSTM prediction models off of it. The predictions of the test model will be stored in `data\out\test_{model}.csv`, which you may verify against the files name `true_test_{model}.csv` to ensure the model is functioning as expected.

## `src\model\model.py`

This file contains the:

- first order Hidden Markov model class that will be used for predicting future foreground applications. After splitting the data and fitting the training set to a `first_order_HMM` instance using `fit`, the model keeps track of the prior and posterior probabilities of the training set's foreground applications. When inputting an observation, `X`, to `predict`, the function returns a list of foregrounds, (of size `n_foregrounds`, with default value of 1) with the highest conditional probability given `X`'s inputted foreground application and the trained model's posterior probabilities. `accuracy` returns the accuracy of the `y_test` on `y_pred` by taking each true foreground application in `y_test` and checking whether or not it appears in its respective list of foregrounds in `y_pred`.

- The next-app prediction LSTM model used a “look-back” value of one previous foreground application in order to predict one future foreground application, where a “look-back” is defined as the number of previous events a single input will use in order to generate the next output

- The duration prediction LSTM used a look-back value of five. In other words, the model uses the previous five data points to predict the next. 

Both LSTMs' model architecture is similar, with the four layers in the same order. 



## Docker
A dockerfile is included and will create a Docker environment that allows for the successful execution of all code in this repository.
