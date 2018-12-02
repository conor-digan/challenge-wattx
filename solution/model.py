from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)



class Data():
    def __init__(self, X_train, X_test, y_train, y_test):
        '''
        Just a class to store the train,test splits easily
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        logging.info('Created the data')



class Model():
    '''
    This is the central class that will train my models
    '''

    def __init__(self, data_df, config):
        '''
        TODO: comment this whole class
        '''

        logging.info('Initializing the model')

        self.base_data = data_df
        self.config = config

        self._generate_data_splits(config['data'])

        logging.info('Training the model')
        self._build_model(config['model'])


    def _generate_data_splits(self, config):

        splits_config = config['splits']

        splits = {}

        for name, params in splits_config.items():

            data = self.base_data

            start_hour = params['start_hour'] if 'start_hour' in params else 0
            end_hour = params['end_hour'] if 'end_hour' in params else np.inf

            rows = (data['hour_slot'] >= start_hour) & (data['hour_slot'] <= end_hour)
            data = data[rows]

            if params['devices'] != 'all':
                rows = data['device'] in params['devices']
                data = data[rows]

            if 'type' in params:
                rows = data['type'] == params['type']
                data = data[rows]

            xy_config = config['xy']
            y = data[xy_config['y_col']]
            X = data[xy_config['X_cols']]

            test_size = params['test_perc']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            data = Data(X_train, X_test, y_train, y_test)

            splits[name] = data

            self.data_splits = splits

    def _build_model(self, model_config):

        model = model_config['model_class']
        model = model(**model_config['paramaters'])

        data = self.data_splits[model_config['data_split']]

        X_train = data.X_train
        X_test = data.X_test

        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        y_train = data.y_train
        y_test = data.y_test

        na_train_rows = X_train.isnull().any(axis=1)
        na_test_rows = X_test.isnull().any(axis=1)

        X_train = X_train[na_train_rows == False]
        y_train = y_train[na_train_rows == False]
        X_test = X_test[na_test_rows == False]
        y_test = y_test[na_test_rows == False]

        data.X_test = X_test
        data.X_train = X_train
        data.y_train = y_train
        data.y_test = y_test

        model.fit(X_train, y_train)

        self.model = model


    def predict(self, data_split = 'predict', use_test=False, predict_proba = False):

        data = self.data_splits[data_split]
        X, y = (data.X_test, data.y_test) if use_test else (data.X_train, data.y_train)
        X = pd.get_dummies(X)

        predicted = self.model.predict_proba(X)[:, 1] if predict_proba else self.model.predict(X)

        predicted = pd.Series(predicted, index=X.index)

        return predicted


    def plot_roc_curve(self, data_split='train', use_test=True):

        data = self.data_splits[data_split]
        X, y = (data.X_test, data.y_test) if use_test else (data.X_train, data.y_train)
        X = pd.get_dummies(X)
        predicted = self.model.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = roc_curve(y, predicted)
        roc_auc = auc(fpr, tpr)
        print('AUC: {}'.format(roc_auc))

        plt.figure()

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()