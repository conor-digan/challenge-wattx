#!/usr/bin/env python3

import itertools
import sys
import logging

import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
from sklearn.ensemble import RandomForestClassifier

from model import Model


logging.basicConfig(level=logging.INFO)


def transform_activation_data(device_activations, starting_time = None):
    '''
    Load in the training data, add in the dates to predict for and transform

    Args:
        device_activations(ps.DataFrame): The activation data to be transformed
        starting_time(timestamp): The timestamp to start the hour_slot from. It should correspond to the earliest time in the training data. If it's not provided then the earliest time in the data is chosen

    Returns:
        object(pd.DataFrame): The transformed training data
    '''

    logging.info('Transforming the activation data')

    device_activations.time = pd.to_datetime(device_activations.time)

    #Extract the required variables from the date
    earliest_date = starting_time if starting_time is not None else min(device_activations.time).date()

    device_activations['date'] = device_activations['time'].dt.date
    device_activations['hour_of_day'] = device_activations['time'].dt.hour
    device_activations['hour_of_week'] = device_activations['time'].dt.dayofweek * 24 + device_activations[
        'hour_of_day']
    device_activations['hour_slot'] = (device_activations['date'] - earliest_date).dt.days * 24 + device_activations[
        'hour_of_day']



    # Create a blank dataset for every hour and device
    all_devices = list(device_activations.device.unique())
    n_hours = ((max(device_activations.time) - min(device_activations.time)).days + 1) * 24
    hour_slot = list(range(n_hours))
    earliest_hour = min(device_activations.hour_slot)
    hour_slot = [x+earliest_hour for x in hour_slot]

    blank_df = pd.DataFrame(list(product(all_devices, hour_slot)), columns=['device', 'hour_slot'])
    blank_df['hour_of_day'] = blank_df.hour_slot % 24
    blank_df['hour_of_week'] = blank_df.hour_slot % (7 * 24)
    blank_df['day_num'] = np.floor(blank_df.hour_slot / 24).astype(int)
    blank_df['week_num'] = np.floor(blank_df.hour_slot / (24 * 7)).astype(int)
    blank_df['day_of_week'] = np.floor(blank_df.hour_of_week / 7).astype(int)



    # Cleanup the activation data in order to join with the blank data
    agg_dict = {
        'time': {
            'earliest_activation': min,
            'latest_activation': max
        },
        'device_activated': sum
    }

    grouped_df = device_activations.groupby(['device', 'hour_slot']).agg(agg_dict).reset_index()
    grouped_df.columns = grouped_df.columns.droplevel(0)
    grouped_df.columns = [
        'device',
        'hour_slot',
        'earliest_activation',
        'latest_activation',
        'total_activations'
    ]


    # Join with the activation data and cleanup
    df = pd.merge(blank_df, grouped_df, how='left')
    df['is_active'] = (df.total_activations.isna() == False).astype(int)


    return df



def create_fake_activation_data(devices, start_time):
    '''
    Create fake activation data in order to use the same transformation functions as the training data

    Args:
        devices(list): The time to start
        start_time(timestamp): The time to start the 24 hours from

    Returns:
        object(pd.DataFrame): The 'fake' activations
    '''


    logging.info('Creating fake activation data for the prediction data')

    next_24_hours = pd.date_range(start_time, periods=24, freq='H').ceil('H')


    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, devices))
    df = pd.DataFrame(xproduct, columns=['time', 'device'])

    #As we're only predicting for 24 hours at a time, this wont affect any feature engineering
    df['device_activated'] = -1
    df.columns = ['time', 'device', 'device_activated']

    return df



def extract_modeling_data(df):
    '''
    Extract the modeling data from the transformed activations

    Args:
        df(ps.DataFrame): The data to extract the modeling data from

    Returns:
        object(pd.DataFrame): The modeling data
    '''


    logging.info('Extracting the modeing data from the transformed activations')

    #Was the device active this time last week ?

    # Select only the required columns
    active_last_week_df = df[['device', 'hour_slot', 'hour_of_week', 'week_num']]

    # Create a lag_week_num variable to join on
    active_last_week_df.loc[:,'lag_week_num'] = active_last_week_df['week_num'] - 1

    # Join with last weeks data
    active_last_week_df = pd.merge(
        active_last_week_df,
        df[['device', 'hour_of_week', 'week_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_week', 'lag_week_num'],
        right_on=['device', 'hour_of_week', 'week_num']
    )

    # Clean up
    active_last_week_df = active_last_week_df[['device', 'hour_slot', 'is_active']]
    active_last_week_df.columns = ['device', 'hour_slot', 'is_active_last_week']



    #Was the device active this time yesterday
    # Select only the required columns
    active_yesterday_df = df[['device', 'hour_slot', 'hour_of_day', 'day_num']]

    # Create a lag_day_num variable to join on
    active_yesterday_df.loc[:,'lag_day_num'] = active_yesterday_df['day_num'] - 1

    # Join with last weeks data
    active_yesterday_df = pd.merge(
        active_yesterday_df,
        df[['device', 'hour_of_day', 'day_num', 'is_active']],
        how='left',
        left_on=['device', 'hour_of_day', 'lag_day_num'],
        right_on=['device', 'hour_of_day', 'day_num']
    )

    # Clean up
    active_yesterday_df = active_yesterday_df[['device', 'hour_slot', 'is_active']]
    active_yesterday_df.columns = ['device', 'hour_slot', 'is_active_yesterday']



    #What sthe average activation rater for this room at this time of the day

    # Sort values by hour_slot
    df.sort_values(by='hour_slot', inplace=True, ascending=True)

    # Select only the required columns
    hour_of_day_activation_rate_df = df[['device', 'hour_slot', 'hour_of_day', 'day_num', 'is_active']]

    # Create a custom aggregation function to calculate the mean before this value
    def mean_pre_now(x):
        return np.mean(x[:-1])

    agg_dict = {
        'is_active': mean_pre_now,
        'hour_slot': max
    }

    # Group by device, hour_of_day
    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df. \
        groupby(['device', 'hour_of_day']). \
        expanding(). \
        agg(agg_dict)[['is_active', 'hour_slot']]. \
        reset_index(drop=False)

    hour_of_day_activation_rate_df = hour_of_day_activation_rate_df[['device', 'hour_slot', 'is_active']]
    hour_of_day_activation_rate_df.columns = ['device', 'hour_slot', 'daily_activation_rate']


    #What's the average activation rater for this room at this time of the week

    # Sort values by hour_slot
    df.sort_values(by='hour_slot', inplace=True, ascending=True)

    # Select only the required columns
    weekly_activation_rate_df = df[['device', 'hour_slot', 'hour_of_week', 'week_num', 'is_active']]

    # Create a custom aggregation function to calculate the mean before this value
    def mean_pre_now(x):
        return np.mean(x[:-1])

    agg_dict = {
        'is_active': mean_pre_now,
        'hour_slot': max
    }

    # Group by device, hour_of_day
    weekly_activation_rate_df = weekly_activation_rate_df. \
        groupby(['device', 'hour_of_week']). \
        expanding(). \
        agg(agg_dict)[['is_active', 'hour_slot']]. \
        reset_index(drop=False)

    weekly_activation_rate_df = weekly_activation_rate_df[['device', 'hour_slot', 'is_active']]
    weekly_activation_rate_df.columns = ['device', 'hour_slot', 'weekly_activation_rate']



    #What's the average activation rate for this room for the last week

    # Get average daily activations for each device
    weekly_device_activation_rate_df = df.groupby(['device', 'day_num']) \
        .agg(np.mean)['is_active'] \
        .reset_index(drop=False)
    weekly_device_activation_rate_df.columns = ['device', 'day_num', 'activation_rate']

    weekly_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_device_activation_rate_df['weeks_activation_rate'] = weekly_device_activation_rate_df \
        .groupby('device')['activation_rate'] \
        .rolling(7).mean() \
        .reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_device_activation_rate_df['lead_day_num'] = weekly_device_activation_rate_df['day_num'] + 1

    weekly_device_activation_rate_df = pd.merge(
        df[['device', 'hour_slot', 'day_num']],
        weekly_device_activation_rate_df,
        how='left',
        left_on=['device', 'day_num'],
        right_on=['device', 'lead_day_num']
    )

    # Clean up
    keep_cols = [
        'device',
        'hour_slot',
        'activation_rate',
        'weeks_activation_rate'
    ]
    weekly_device_activation_rate_df = weekly_device_activation_rate_df[keep_cols]
    weekly_device_activation_rate_df.columns = [
        'device',
        'hour_slot',
        'yesterdays_device_activation_rate',
        'last_weeks_device_activation_rate'
    ]


    #What's the average activation rate for all rooms for the last week

    # Get average daily activations for each device
    weekly_all_device_activation_rate_df = df.groupby('day_num') \
        .agg(np.mean)['is_active'] \
        .reset_index(drop=False)
    weekly_all_device_activation_rate_df.columns = ['day_num', 'activation_rate']

    weekly_all_device_activation_rate_df.sort_values('day_num', ascending=True)
    weekly_all_device_activation_rate_df['weeks_activation_rate'] = weekly_all_device_activation_rate_df[
        'activation_rate'] \
        .rolling(7).mean() \
        .reset_index(drop=True)

    # Add 1 to the day for joining with original df
    weekly_all_device_activation_rate_df['lead_day_num'] = weekly_all_device_activation_rate_df['day_num'] + 1

    weekly_all_device_activation_rate_df = pd.merge(
        df[['device', 'hour_slot', 'day_num']],
        weekly_all_device_activation_rate_df,
        how='left',
        left_on='day_num',
        right_on='lead_day_num'
    )

    # Clean up
    keep_cols = [
        'device',
        'hour_slot',
        'activation_rate',
        'weeks_activation_rate'
    ]
    weekly_all_device_activation_rate_df = weekly_all_device_activation_rate_df[keep_cols]
    weekly_all_device_activation_rate_df.columns = [
        'device',
        'hour_slot',
        'yesterdays_all_device_activation_rate',
        'last_weeks_all_device_activation_rate'
    ]



    #Join all of the dataframes into one single modeling dataset

    # Add in whether the device was active this time last week
    modeling_df = pd.merge(
        df,
        active_last_week_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )

    # Add in whether the device was active this time yesterday
    modeling_df = pd.merge(
        modeling_df,
        active_yesterday_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )

    # Add in average previous activation rate for this time of day
    modeling_df = pd.merge(
        modeling_df,
        hour_of_day_activation_rate_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )

    # Add in average previous activation rate for this time of day & day of week
    modeling_df = pd.merge(
        modeling_df,
        weekly_activation_rate_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )

    # Add in average previous activation rate for this device for yesterday & last week
    modeling_df = pd.merge(
        modeling_df,
        weekly_device_activation_rate_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )

    # Add in average previous activation rate for all device for yesterday & last week
    modeling_df = pd.merge(
        modeling_df,
        weekly_all_device_activation_rate_df,
        how='left',
        left_on=['device', 'hour_slot'],
        right_on=['device', 'hour_slot']
    )


    return modeling_df



def get_config(pred_data_start_hour):
    '''
    Get the config for the model
    Note: This

    Args:
        pred_data_start_hour: The cutoff between the train & predict datasets

    Returns:
        dictionary: The configuration to input to the model
    '''


    logging.info('Getting the config')

    config = {
        'data': {
            'splits': {
                'train': {
                    'test_perc': 0,
                    'start_hour': 168,
                    'devices': 'all',
                    'type': 'train'
                },
                'predict': {
                    'test_perc': 0,
                    'start_hour': pred_data_start_hour,
                    'devices': 'all',
                    'type': 'predict'
                }
            },
            'xy': {
                'y_col': 'is_active',
                'X_cols': [
                    'device',
                    'hour_of_day',
                    'hour_of_week',
                    'day_of_week',
                    'daily_activation_rate',
                    'weekly_activation_rate'
                ]
            }
        },
        'model': {
            'data_split': 'train',
            'model_class': RandomForestClassifier,
            'paramaters': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'max_features': 'auto'
            }
        }
    }

    return config



def build_model(data, config):
    '''
    Initialize & train the model

    Args:
        data: The modeling data to build with
        config: The modeling configuration

    Returns:
        object(Model): The trained model class
    '''

    logging.info('Building the model')

    model = Model(data, config)
    return model





if __name__ == '__main__':


    #Extract the arguments
    pred_time, in_file, out_file = sys.argv[1:]

    #Load the activation data and format the time correctly
    device_activations = pd.read_csv(in_file)
    device_activations.time = pd.to_datetime(device_activations.time)

    logging.info('Read in the activation data. Shape: {}'.format(device_activations.shape))

    #Get some variables to be used later
    starting_time = min(device_activations.time).date()
    devices = list(device_activations['device'].unique())

    #Create the training & prediction data
    train_data = transform_activation_data(device_activations)
    predict_data = transform_activation_data(create_fake_activation_data(devices, pred_time),starting_time=starting_time)
    train_data['type'] = 'train'
    predict_data['type'] = 'predict'
    pred_data_start_hour = min(predict_data['hour_slot'])

    logging.info('Created the training & prediction datasets')

    #Append the datasets together
    all_data = train_data.append(predict_data)

    #Create the modeling dataset
    modeling_data = extract_modeling_data(all_data)

    logging.info('Created the modeling dataset')

    #Get the config for the model
    config = get_config(pred_data_start_hour)

    #Build & Train the model
    model = build_model(modeling_data, config)

    logging.info('Built & trained the model')

    #Predict for the 'predict' data
    preds = model.predict()

    logging.info('Predicted for the desired day')

    #Create the final dataframe
    results_data = pd.DataFrame(preds).join(
        modeling_data[['device', 'hour_slot']],
        how = 'left'
    )

    #Clean up the final dataframe and write to csv at the specified location
    def convert_hour_slot_to_ts(hour_slot):
        return starting_time + dt.timedelta(hours = hour_slot)

    results_data['hour_slot'] = results_data['hour_slot'].apply(convert_hour_slot_to_ts)

    results_data.columns = [
        'activation_predicted',
        'device',
        'time'
    ]

    results_data.sort_values(by = ['device', 'time'])
    results_data.to_csv(out_file, index=False)

    logging.info('Saved the results to: {}'.format(out_file))
