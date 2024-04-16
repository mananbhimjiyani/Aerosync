from django.shortcuts import render
from datetime import datetime
from joblib import load
import numpy as np
import logging
import os
import math


# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG


base_dir = os.path.dirname(__file__)


# our home page view
def home(request):
    label_mapping = get_label_mapping('Validating Airline Code')  # Get the label mapping
    origin_mapping = get_label_mapping('Origin')
    destination_mapping = get_label_mapping('Destination')
    return render(request, 'index.html', {'label_mapping': label_mapping, 'origin_mapping': origin_mapping,
                                          'destination_mapping': destination_mapping})


def getFlightDelayPrediction(origin, destination, airline_code, year, month, day, month_sin, month_cos, day_sin,
                             day_cos):
    # Get the base directory path

    # Construct the full path to the model file
    model_path = os.path.join(base_dir, "decision_tree_model_price.joblib")

    # Load the pre-trained decision tree model without standardization
    model = load(model_path)

    # Prepare the input data
    manual_input = [origin, destination, airline_code, year, month, day, month_sin, month_cos, day_sin, day_cos]
    manual_input_reshaped = np.array(manual_input).reshape(1, -1)

    # Make prediction
    prediction = model.predict(manual_input_reshaped)

    # Return prediction
    return prediction[0]


# Example usage:
prediction = getFlightDelayPrediction(27, 22, 12, 2024, 6, 15, 0.5, -0.866, 0.258, -0.9659)
print(prediction)  # This will print the predicted output


def is_leap_year(year):
    if year % 4 == 0:
        if year % 100 != 0 or year % 400 == 0:
            return True
    return False


def days_in_month(month, year):
    if month in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    elif month in {4, 6, 9, 11}:
        return 30
    elif month == 2:
        if is_leap_year(year):
            return 29
        else:
            return 28


def convert_to_sin_cos(day, month, year):
    # Calculate the total number of days in the year
    days_in_year = 365 if not is_leap_year(year) else 366

    # Convert days to radians
    day_rad = (day - 1) * (2 * math.pi / days_in_month(month, year))

    # Convert months to radians
    month_rad = (month - 1) * (2 * math.pi / 12)

    # Calculate sine and cosine values
    day_sin = math.sin(day_rad)
    day_cos = math.cos(day_rad)
    month_sin = math.sin(month_rad)
    month_cos = math.cos(month_rad)

    return day_sin, day_cos, month_sin, month_cos


import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_label_mapping(column_name):
    # Read the CSV file
    df = pd.read_csv(os.path.join(base_dir, 'flight_data_price_1.csv'))

    # Label encode the specified column
    label_encoder = LabelEncoder()
    df[f'{column_name} Encoded'] = label_encoder.fit_transform(df[column_name])

    # Create mapping of encoded values to original strings
    label_mapping = {label: value.strip("[]'") for label, value in
                     zip(label_encoder.transform(df[column_name]), df[column_name])}

    return label_mapping


# our result page view
def result(request):
    origin = int(request.GET['origin'])
    destination = int(request.GET['destination'])
    airline_code = int(request.GET['airline_code'])

    # Parse date
    date_str = request.GET['date']
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    day_sin, day_cos, month_sin, month_cos = convert_to_sin_cos(day, month, year)


    result = getFlightDelayPrediction(origin, destination, airline_code, year, month, day, month_sin, month_cos,
                                      day_sin, day_cos)
    # Convert the result to a human-readable format
    if result > 0:
        delay_minutes = abs(result) * 60
        result_text = f"{delay_minutes:.1f} minutes early"
    else:
        delay_minutes = abs(result) * 60
        result_text = f"{delay_minutes:.1f} minutes late"

    # Pass the result_text variable to the template
    return render(request, 'result.html', {'result': result_text})

