import sys
import os

import pandas as pd
import numpy as np

import sklearn
import xgboost as xgb

# Write test function to assert?

print(sys.version)
print("Pandas Version: " + pd.__version__)
print("Numpy Version: " + np.__version__)
print("XGBoost Version: " + xgb.__version__)
print("scikit-learn Version: " + sklearn.__version__)

'''
wrapper function for data loading; test for nulls!
'''


def data_load(flow, creek, weather):

    # Load influent flow data
    flow = pd.read_csv(
        'influent_flow.csv',
        parse_dates=['Date']  # ,
        # infer_datetime_format=True
    )

    # Load creek data
    creek = pd.read_table(
        'usgs_san_francisquito.txt',
        sep='\t',
        engine='python',
        skiprows=lambda x: x in range(0, 30),
        header=0
        # lineterminator='\r\n'
    )

    # Load weather data

    weather = pd.read_csv(
        'weather.csv',
        parse_dates=['Date']
    )


def data_wrangle(creek):
    creek = creek.dropna()

    # Drop column-definition row.
    creek = creek.drop(0)

    # Format datetime column and float datatypes, and rename columns.
    creek['Date'] = pd.to_datetime(
        creek['datetime'], format='%Y-%m-%d %H:%M:%S')
    creek = creek.astype({'14747_00060': 'float64', '14748_00065': 'float64'})
    creek = creek.rename(
        columns={'14747_00060': 'discharge_cfs', '14748_00065': 'gauge_height_ft'})

    # Resample data by hour and max value
    creek = creek.set_index('Date').resample(
        'H')['discharge_cfs', 'gauge_height_ft'].max().reset_index()


def data_join(train, df1=data_load.flow, df2=data_load.weather, df3=data_wrangle.creek):
    '''
    #Merge outer and test for nulls!
    '''

    if train == True:
        # Merge dataframes
        dataset = df1.merge(df2, on='Date', how='outer')
        dataset = dataset.merge(df3, on='Date', how='inner')

    else:
        dataset = df2.merge(df3, on='Date', how='inner')


def feature_extraction(dataset):
    '''
    Time series munging
    '''

    # Create day of week, month, and hour integer columns
    dataset['month'] = dataset['Date'].dt.month
    dataset['hour'] = dataset['Date'].dt.hour + 1

    # Create unit circle coordinates for hour
    dataset['hour_x'] = np.sin(2.*np.pi*(dataset.hour/24))
    dataset['hour_y'] = np.cos(2.*np.pi*(dataset.hour/24))

    # Create unit circle coordinates for month
    dataset['month_x'] = np.sin(2.*np.pi*(dataset.month/12))
    dataset['month_y'] = np.cos(2.*np.pi*(dataset.month/12))

def create_validation_set(dataset):
    # Create validation dataset
    # Filter by date to create separate dataframes for train/testing and validation
    dataset = dataset[dataset['Date'] >= '2013-01-15 00:00:00']

    validation = dataset[
        (dataset['Date'] > '2016-12-22 23:00:00') &
        (dataset['Date'] <= '2018-01-24 07:00:00')
    ]

    pd.to_csv()

    dataset = dataset[dataset['Date'] <= '2016-12-22 23:00:00']

def model_train(data):
    '''
    Need model training functions!
    '''
    # Split into train/test data
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.25)

    # Create separate dataframes for features and target variable
    Xtrain = train.drop(columns=['Date', 'INF_FLOW'], axis=1)
    Xtest = test.drop(columns=['Date', 'INF_FLOW'], axis=1)

    Ytrain = train[['INF_FLOW']]
    Ytest = test[['INF_FLOW']]

    # Load these dataframes into XGBoost DMatrices for training and evaluation
    Dtrain = xgb.DMatrix(Xtrain, Ytrain)
    Dtest = xgb.DMatrix(Xtest, Ytest)

    model.save_model('flow_model.json')

def model_predict():
    # Create XGB DMatrix from Validation Dataframe
    val_df = validation.drop(['Date'], axis=1)
    val_df = xgb.DMatrix(val_df)

    model.save_model('flow_model.json')

    # Run predictions on Validation set
    predictions = model.predict(val_df)
    validation['predictions'] = predictions

    # Add residuals
    interval = validation['predictions'] + 3
