'''
Contains code for the First Order Hidden Markov Model
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict


# +
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


# -

def build(string_df, test_size, outpath):
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

    # saving predictions to .txt file
    df = pd.DataFrame({'X': X_test, 'y_pred': y_pred})
    df.to_csv(outpath, header=True, index=None, sep=',', mode='w')
