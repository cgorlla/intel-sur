"""
Contains code for building all models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict


def build_hmm(string_df, test_size, outpath, test):
    """
    Contains code for the First Order Hidden Markov Model
    """
    def clean(df, condition_1=True):

        sequence = df[df['ID_INPUT'] == 3]['VALUE'].values
        pairs = []

        for i in range(len(sequence)-1):
            if condition_1 == True:
                if sequence[i] == sequence[i+1]:
                    continue
                else:
                    pairs.append([sequence[i], sequence[i+1]])
            else:
                pairs.append([sequence[i], sequence[i+1]])

        return pd.DataFrame(pairs, columns=['X', 'y'])

    class first_order_HMM(object):

        def __init__(self, string_df, computer=0):
            self.uniques = sorted(string_df[string_df['ID_INPUT'] == 3]['VALUE'].unique())
            self.data = pd.DataFrame()
            self.counts = defaultdict(float)
            self.priors = defaultdict(float)
            self.posteriors = defaultdict(float)

        def fit(self, X, y):
            """
            inputs :
                X : a list of prior foreground applications
                y : a list of subsequent foreground applications from the prior foreground applications

            outputs:
                None
            """

            self.data = pd.DataFrame({'X': X, 'y': y})

            def get_counts():
                counts = defaultdict(float)
                for foreground in self.data.values:
                    counts[foreground[0]] += 1
                counts[self.data.values[-1][1]] += 1
                return pd.Series(dict(sorted(counts.items())))

            def get_priors():
                priors = defaultdict(float)
                # getting percentage of each unique foreground's occurrence
                for foreground, count in self.counts.items():
                    priors[foreground] = count / (len(self.data.values) + 1)
                return pd.Series(priors)

            def get_posteriors():
                # creating empty conditional probability matrix
                posteriors = pd.DataFrame(
                    np.zeros([len(self.uniques), len(self.uniques)]), 
                    index=self.uniques, 
                    columns=self.uniques
                )           
                # counting pairs of foregrounds
                for pair in self.data.values:
                    posteriors.loc[pair[0], pair[1]] += 1
                # calculating conditional probability of foreground A given foreground B
                posteriors = posteriors.apply(lambda x: x / sum(x), axis=1)
                return posteriors

            self.counts = get_counts()
            self.priors = get_priors()
            self.posteriors = get_posteriors()

        def predict(self, X, n_foregrounds=1):
            """
            inputs :
                X : a list of prior foreground applications
                n_foregrounds : number of predicted foregrounds to return (default: 1)

            outputs :
                y : a list of predicted subsequent foreground applications 
            """

            # outputting foreground application with maximum conditional probability
            y = []
            for x in X:
                # outputting foreground application with maximum conditional probability
                # y = np.append(y, self.posteriors.loc[x,:].idxmax())
                y.append(list(self.posteriors.loc[x,:].sort_values(ascending=False)[:n_foregrounds].index))

            return y

        def accuracy(self, y_test, y_pred):
            """
            inputs :
                y_test : a list of true subsequent foreground applications
                y_pred : a list of predicted subsequent foreground applications

            outputs :
                accuracy : accuracy of trained model on y_test
            """
            correct = 0
            for i, y in enumerate(y_test):
                if y in y_pred[i]:
                    correct += 1
            accuracy = correct / len(y_test)

            return accuracy
    print('HIDDEN MARKOV MODEL')
    print('Cleaning data...')
    df = clean(string_df)

    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'], test_size = test_size)

    # initializing first order HMM
    model = first_order_HMM(string_df)

    print('Training model...')
    # training model
    model.fit(X_train, y_train)

    # computing test accuracy
    y_pred = model.predict(X_test, n_foregrounds=3)
    accuracy = model.accuracy(y_test, y_pred)
    print('Test accuracy: ', accuracy)
    print()
    if test:
        outpath = outpath + 'test'
    # saving predictions to .txt file
    df = pd.DataFrame({'X': X_test, 'y_pred': y_pred})
    df.to_csv(outpath + '_hmm_next.csv', header=True, index=None, sep=',', mode='w')


def build_lstm_duration(string_df, test_size, outpath, test):
    
    print("LSTM DURATION MODEL")
    
    print('Cleaning data...')
    
    df_0 = string_df[string_df['ID_INPUT'] == 3].drop(['ID_INPUT', 'PRIVATE_DATA'], axis=1).reset_index(drop=True)

    # converting 'MEASUREMENT_TIME' column to datetime
    df_0.loc[:, 'MEASUREMENT_TIME'] = pd.to_datetime(df_0['MEASUREMENT_TIME'])

    # using converted datetime column to get usage per application ('TIME_DELTA')
    time_delta = (df_0['MEASUREMENT_TIME'].shift(periods=-1) - df_0['MEASUREMENT_TIME']).drop(len(df_0)-1).apply(lambda x: float(x.total_seconds() / 60))
    time_delta = time_delta.append(pd.Series(-1), ignore_index=True)

    # getting usage per future application ('TIME_DELTA_1')
    time_delta_1 = time_delta.shift(-1)

    # adding 'TIME_DELTA' and 'TIME_DELTA_1' to DataFrame, dropping last instances with no values, and converting outliers (large numbers) to 60 minute values
    df_0 = df_0.assign(**{'TIME_DELTA': time_delta, 'TIME_DELTA_1': time_delta_1})
    df_0 = df_0.drop([len(df_0)-2, len(df_0)-1]).drop(['MEASUREMENT_TIME', 'VALUE'], axis=1)
    df_0 = df_0.applymap(lambda x: 60 if x > 60 else x)
    
    df_0['MEASUREMENT_TIME_HR'] = pd.to_datetime(string_df['MEASUREMENT_TIME']).dt.hour
    df_0['MEASUREMENT_TIME'] = (string_df['MEASUREMENT_TIME'])
    df_0['VALUE'] = string_df['VALUE']
    
    print('Splitting data...')
    
    X_train, X_test, y_train, y_test = train_test_split(df_0[['TIME_DELTA','MEASUREMENT_TIME_HR']], df_0['TIME_DELTA_1'], test_size=test_size, shuffle=False)
    X_train1, X_test1, null1, null2 = train_test_split(pd.get_dummies(df_0['VALUE']), df_0['TIME_DELTA_1'], test_size=test_size, shuffle=False)
    X_train = X_train.join(X_train1)
    X_test = X_test.join(X_test1)
    
    model = keras.Sequential()
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dense(units=1, activation='relu'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanAbsoluteError())
    
    print('Training model...')
    model.fit(x=X_train.to_numpy().reshape(-1,1 ,X_train.shape[1]), y= y_train, epochs= 23, validation_data = (X_test.to_numpy().reshape(-1,1 ,X_train.shape[1]),(y_test)))
    print('Predicting next apps...')
    preds = model.predict(X_train.append(X_test).to_numpy().reshape(-1,1 ,X_train.shape[1]))
    
    lstm_out = pd.DataFrame()
    lstm_out['App'] = df_0['VALUE']
    lstm_out['Predicted Usage Duration'] = [x[0] for x in preds]
    if test:
        outpath = outpath + 'test'
        
    print('Saving outputs...')
    lstm_out.to_csv(outpath + '_lstm_dur.csv', header=True, index=None, sep=',', mode='w')


def build_lstm_next(string_df, test_size, outpath, test):
    print("LSTM NEXT-APP MODEL")
    
    print('Cleaning data...')
    temp = string_df.copy()
    temp = temp[temp['ID_INPUT'] == 3][['MEASUREMENT_TIME', 'VALUE']].reset_index(drop=True)
    temp.loc[:, 'MEASUREMENT_TIME'] = pd.to_datetime(temp['MEASUREMENT_TIME'])
    time_delta = (temp['MEASUREMENT_TIME'].shift(periods=-1) - temp['MEASUREMENT_TIME']).drop(len(temp)-1, axis=0).apply(lambda x: float(x.total_seconds() / 60))
    time_delta = time_delta.append(pd.Series(-1), ignore_index=True)
    hour = temp['MEASUREMENT_TIME'].apply(lambda x: x.hour)
    temp = temp.assign(**{'TIME_DELTA': time_delta, 'MEASUREMENT_TIME_HOUR_ONLY': hour})
    cols = temp['VALUE'].value_counts().index.to_list()
    cols.extend(['MEASUREMENT_TYPE', 'VALUE', 'TIME_DELTA', 'MEASUREMENT_TIME_HOUR_ONLY'])
    temp['VALUE'].loc[~temp['VALUE'].isin(cols)] = "other"
    temp['VALUE_SHIFT'] = temp['VALUE'].shift(periods=-1, axis=0)
    temp = temp.drop(len(temp)-1)
    
    #Label encoding values
    label_encoder = LabelEncoder()
    temp['VALUE_SHIFT'] = label_encoder.fit_transform(temp['VALUE_SHIFT'])
    onehot = pd.get_dummies(temp['VALUE'])
    temp = pd.concat([temp, onehot], axis=1)
    temp = temp.drop(['VALUE', 'MEASUREMENT_TIME'], axis=1)
    labels = temp['VALUE_SHIFT'].values
    
    print('Splitting data...')
    
    #Prepare next app data for LSTM input format
    X_train_df = temp.drop(columns=['VALUE_SHIFT'])
    X_train, X_test, y_train, y_test = train_test_split(X_train_df.values, labels, test_size=test_size, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    y_train = y_train.to_numpy().reshape((y_train.shape[0], 1, y_train.shape[1]))
    y_test = y_test.to_numpy().reshape((y_test.shape[0], 1, y_test.shape[1]))
    
    
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print('Training model...')
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split=0.1, shuffle=False)
    y_pred = model.predict(X_test)
    
    print('Predicting next apps...')
    test = []
    for i in y_pred:
        test.append(np.argsort(i[0])[-4:][::-1])
    y_pred2 = pd.DataFrame(test)
    test = []
    for i in y_pred:
        test.append(i[0])
    y_test2 = pd.DataFrame(test)
    y_test2 = y_test2.idxmax(axis=1)

    compare = pd.DataFrame()
    compare['Actual'] = y_test2
    compare = pd.concat([compare, y_pred2], axis=1)
    compare['0_true'] = (compare['Actual'] == compare[0])
    compare['1_true'] = (compare['Actual'] == compare[1])
    compare['Accurate'] = (compare['0_true'] | compare['1_true'])
    compare = compare[[0,1]]
    compare[0] = label_encoder.inverse_transform(compare[0])
    compare[1] = label_encoder.inverse_transform(compare[1])
    
    print('Saving outputs...')
    # saving predictions to .csv file
    output = pd.DataFrame(y_pred.flatten(), columns=['y_pred'])
    if test:
        outpath = outpath + 'test'
    output.to_csv(outpath+'_lstm_next.csv', header=True, index=None, sep=',', mode='w')


def build(string_df, test, test_size, outpath):
    print('Next-App Prediction Engine Intializing...')
    # Next-App HMM 
    build_hmm(string_df, test_size, outpath, test)
    # Next-App LSTM
    build_lstm_next(string_df, test_size, outpath, test)
    print('App Duration Prediction Engine Initializing...')
    # Duration LSTM
    build_lstm_duration(string_df, test_size, outpath, test)
