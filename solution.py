#!/usr/bin/env python3

import itertools
import numpy as np
import pandas as pd
import sys


def predict_future_activation(current_time, previous_readings):
    """This function predicts future hourly activation given previous sensings.
    It's probably not the best implementation as it just returns a random
    guess.
    """
    # make predictable
    np.random.seed(len(previous_readings))

    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')

    device_names = sorted(previous_readings.device.unique())

    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    predictions.set_index('time', inplace=True)

    # Random guess!
    predictions['activation_predicted'] = np.random.randint(2, size=len(predictions))
    return predictions


def load_and_transform_training_data(in_file):
    '''
    
    '''


if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    previous_readings = pd.read_csv(in_file)
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)