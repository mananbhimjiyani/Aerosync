import requests
import json
import logging
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime
from django.shortcuts import render
from joblib import load
from sklearn.preprocessing import LabelEncoder
from django.test import RequestFactory
from django.http import JsonResponse

# Set up logging configuration (Optional)
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

base_dir = os.path.dirname(__file__)

origin_city_name_df_csv_final = os.path.join(base_dir, 'csv/encoded_unique_data.csv')
dest_city_name_df_csv_final = os.path.join(base_dir, 'csv/encoded_unique_data_dest.csv')

origin_city_name_df_final = pd.read_csv(origin_city_name_df_csv_final)
dest_city_name_df_final = pd.read_csv(dest_city_name_df_csv_final)


# our home page view
def home(request):
    return render(request, 'index.html', )


def chatbot(request):
    return render(request, 'chatbot.html', )


def flight_fare_prediction(request):
    label_mapping = get_label_mapping('Validating Airline Code')  # Get the label mapping
    origin_mapping = get_label_mapping('Origin')
    destination_mapping = get_label_mapping('Destination')
    return render(request, 'fare.html', {'label_mapping': label_mapping, 'origin_mapping': origin_mapping,
                                         'destination_mapping': destination_mapping})


def flight_delay_prediction(request):
    return render(request, 'delay.html')


def getData(name):
    csv_path = None
    if name == 'Origin':
        csv_path = os.path.join(base_dir, 'csv/encoded_unique_data.csv')
    elif name == 'Dest':
        csv_path = os.path.join(base_dir, 'csv/encoded_unique_data_dest.csv')
    elif name == 'Airline':
        csv_path = os.path.join(base_dir, 'csv/airline_df.csv')
    elif name == 'OriginCityName':
        csv_path = os.path.join(base_dir, 'csv/origin_city_name_df.csv')
    elif name == 'DestCityName':
        csv_path = os.path.join(base_dir, 'csv/dest_city_name_df.csv')

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        return df.values.tolist()
    else:
        return None  # Return None if csv_path is not set


def flight_arrival_delay(request):
    delay_origin = getData('Origin')
    delay_destination = getData('Dest')
    delay_airline_code = getData('Airline')
    delay_origin_city_name = getData('OriginCityName')
    delay_destination_city_name = getData('DestCityName')
    return render(request, 'arrival_delay.html', {'delay_origin': delay_origin, 'delay_destination': delay_destination,
                                                  'delay_origin_city_name': delay_origin_city_name,
                                                  'delay_destination_city_name': delay_destination_city_name,
                                                  'airline_code': delay_airline_code})


def flight_arrival_delay_result(request):
    global resultStatement
    origin = int(request.GET['origin'])
    dest = int(request.GET['destination'])
    airline_code = int(request.GET['airline_code'])
    departure_date = request.GET['departure_date']
    departure_time = request.GET['departure_time']
    departureTime = departure_time.replace(':', '')
    departureTime = int(departureTime)
    arrival_date = request.GET['departure_date']
    arrival_time = request.GET['arrival_time']
    arrivalTime = departure_time.replace(':', '')
    arrivalTime = int(arrivalTime)
    diverted = bool(request.GET['diverted'])

    # Parse arrival time
    datetime_obj = datetime.strptime(arrival_date, '%Y-%m-%d')

    # Parse departure time
    datetime_obj = datetime.strptime(departure_date, '%Y-%m-%d')
    dept_day = datetime_obj.day
    dept_month = datetime_obj.month
    dept_year = datetime_obj.year

    departure_day_sin, departure_day_cos, departure_month_sin, departure_month_cos, departure_week_sin, departure_week_cos = convert_to_sin_cos(
        dept_day,
        dept_month,
        dept_year)

    crsarrtime = get_airtime(departure_date, departure_time, arrival_date, arrival_time)

    originairportid, originCityName = get_origin_airport_id(origin_city_name_df_final, origin)
    destairportid, destCityName = get_origin_airport_id(dest_city_name_df_final, dest, )

    result = getFlightArrivalDelayPrediction(dept_year, airline_code, origin, dest, diverted, departureTime,
                                             arrivalTime,
                                             departure_month_sin, departure_month_cos, departure_week_sin,
                                             departure_week_cos, originairportid, originCityName, destairportid,
                                             destCityName, crsarrtime)

    # Check if the flight delay exceeds 10 minutes
    if result > 10:
        resultStatement = f"The flight is expected to experience a delay of approximately {round(abs(result[0]), 2)} minutes."

    # Check if the flight is expected to depart earlier than scheduled by more than 10 minutes
    elif result < -10:
        resultStatement = f"The flight is anticipated to depart earlier than expected by approximately {round(abs(result[0]), 2)} minutes."

    # Check if the flight is expected to depart within a 10-minute window of the scheduled time
    elif -10 < result < 10:
        resultStatement = "The flight is likely to depart on time, with no significant delay anticipated."

    return render(request, 'delay_result.html', {'result': resultStatement})


def flight_departure_delay_result(request):
    global resultStatement
    origin = int(request.GET['origin'])
    dest = int(request.GET['destination'])
    airline_code = int(request.GET['airline_code'])
    departure_date = request.GET['departure_date']
    departure_time = request.GET['departure_time']
    departureTime = departure_time.replace(':', '')
    departureTime = int(departureTime)
    arrival_date = request.GET['departure_date']
    arrival_time = request.GET['arrival_time']
    arrivalTime = departure_time.replace(':', '')
    arrivalTime = int(arrivalTime)
    diverted = bool(request.GET['diverted'])

    # Parse arrival time
    datetime_obj = datetime.strptime(arrival_date, '%Y-%m-%d')
    arr_day = datetime_obj.day
    arr_month = datetime_obj.month
    arr_year = datetime_obj.year

    # Parse departure time
    datetime_obj = datetime.strptime(departure_date, '%Y-%m-%d')
    dept_day = datetime_obj.day
    dept_month = datetime_obj.month
    dept_year = datetime_obj.year

    departure_day_sin, departure_day_cos, departure_month_sin, departure_month_cos, departure_week_sin, departure_week_cos = convert_to_sin_cos(
        dept_day,
        dept_month,
        dept_year)

    crsarrtime = get_airtime(departure_date, departure_time, arrival_date, arrival_time)

    originairportid, originCityName = get_origin_airport_id(origin_city_name_df_final, origin)
    destairportid, destCityName = get_origin_airport_id(dest_city_name_df_final, dest, )

    result = getFlightDepartureDelayPrediction(dept_year, airline_code, origin, dest, diverted, departureTime,
                                               arrivalTime,
                                               departure_month_sin, departure_month_cos, departure_week_sin,
                                               departure_week_cos, originairportid, originCityName, destairportid,
                                               destCityName, crsarrtime)

    # Check if the flight delay exceeds 10 minutes
    if result > 10:
        resultStatement = f"The flight is expected to experience a delay of approximately {round(abs(result[0]), 2)} minutes."

    # Check if the flight is expected to depart earlier than scheduled by more than 10 minutes
    elif result < -10:
        resultStatement = f"The flight is anticipated to depart earlier than expected by approximately {round(abs(result[0]), 2)} minutes."

    # Check if the flight is expected to depart within a 10-minute window of the scheduled time
    elif -10 < result < 10:
        resultStatement = "The flight is likely to depart on time, with no significant delay anticipated."

    return render(request, 'delay_result.html', {'result': resultStatement})


def get_airtime(departure_date, departure_time, arrival_date, arrival_time):
    # Combine departure date and time
    departure_datetime = datetime.strptime(departure_date + ' ' + departure_time, '%Y-%m-%d %H:%M')

    # Combine arrival date and time
    arrival_datetime = datetime.strptime(arrival_date + ' ' + arrival_time, '%Y-%m-%d %H:%M')

    # Calculate the airtime in minutes
    airtime_seconds = (arrival_datetime - departure_datetime).total_seconds()
    airtime_minutes = airtime_seconds / 60

    return int(airtime_minutes)


def map_destinations(dest_city_counter, dest_city_name_df):
    city = None  # Initialize city variable to None
    for i, city_name in enumerate(dest_city_name_df.iloc[:, 0], start=1):
        if i == 1 and city_name == 0:
            continue  # Skip the first row if it contains 0
        if i >= dest_city_counter:
            city = city_name  # Store the current city name in the city variable
            break
    return city  # Return the city name


def flight_departure_delay(request):
    delay_origin = getData('Origin')
    delay_destination = getData('Dest')
    delay_airline_code = getData('Airline')
    delay_origin_city_name = getData('OriginCityName')
    delay_destination_city_name = getData('DestCityName')
    return render(request, 'departure_delay.html',
                  {'delay_origin': delay_origin, 'delay_destination': delay_destination,
                   'delay_origin_city_name': delay_origin_city_name,
                   'delay_destination_city_name': delay_destination_city_name,
                   'airline_code': delay_airline_code})


def getFlightfarePrediction(origin, destination, airline_code, year, month, day, month_sin, month_cos, day_sin,
                            day_cos):
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


def getFlightArrivalDelayPrediction(year, airline_code, origin, dest, diverted, deptime, arrtime, month_sin, month_cos,
                                    week_sin, week_cos, originairportid, origincityname, destairportid, destcityname,
                                    crsarrtime):
    # Construct the full path to the model file
    model_path = os.path.join(base_dir, "linear_regression_model.joblib")

    # Load the pre-trained decision tree model without standardization
    model = load(model_path)

    # Prepare the input data
    manual_input = [[year, airline_code, origin, dest, diverted, deptime, arrtime, month_sin, month_cos,
                     week_sin, week_cos, originairportid, origincityname, destairportid, destcityname,
                     crsarrtime]]
    predicted_delay = model.predict(manual_input)

    # Return prediction
    return predicted_delay


def getFlightDepartureDelayPrediction(year, airline_code, origin, dest, diverted, deptime, arrtime, month_sin,
                                      month_cos,
                                      week_sin, week_cos, originairportid, origincityname, destairportid, destcityname,
                                      crsarrtime):
    # Construct the full path to the model file
    model_path = os.path.join(base_dir, "departure_linear_regression_model.pkl")

    # Load the pre-trained decision tree model without standardization
    model = load(model_path)

    # Prepare the input data
    manual_input = [[year, airline_code, origin, dest, diverted, deptime, arrtime, month_sin, month_cos,
                     week_sin, week_cos, originairportid, origincityname, destairportid, destcityname,
                     crsarrtime]]
    manual_input_reshaped = np.array(manual_input).reshape(1, -1)

    # Make prediction
    prediction = model.predict(manual_input_reshaped)

    # Return prediction
    return prediction


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

    # Calculate week
    week = (day + (month - 1) * days_in_month(month, year)) // 7

    # Convert week to radians
    week_rad = week * (2 * math.pi / (days_in_year // 7))

    # Calculate sine and cosine values
    day_sin = math.sin(day_rad)
    day_cos = math.cos(day_rad)
    month_sin = math.sin(month_rad)
    month_cos = math.cos(month_rad)
    week_sin = math.sin(week_rad)
    week_cos = math.cos(week_rad)

    return day_sin, day_cos, month_sin, month_cos, week_sin, week_cos


def get_label_mapping(column_name):
    # Read the CSV file
    df = pd.read_csv(os.path.join(base_dir, 'csv/flight_data_price_1.csv'))

    # Label encode the specified column
    label_encoder = LabelEncoder()
    df[f'{column_name} Encoded'] = label_encoder.fit_transform(df[column_name])

    # Create mapping of encoded values to original strings
    label_mapping = {label: value.strip("[]'") for label, value in
                     zip(label_encoder.transform(df[column_name]), df[column_name])}

    return label_mapping


# our result page view
def fare_result(request):
    origin = int(request.GET['origin'])
    destination = int(request.GET['destination'])
    airline_code = int(request.GET['airline_code'])

    # Parse date
    date_str = request.GET['date']
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    day_sin, day_cos, month_sin, month_cos, week_sin, week_cos = convert_to_sin_cos(day, month, year)

    result = getFlightfarePrediction(origin, destination, airline_code, year, month, day, month_sin, month_cos,
                                     day_sin, day_cos)

    result_text = round(abs(result) * 100, 2)

    # Pass the result_text variable to the template
    return render(request, 'fare_result.html', {'result': result_text})


def get_origin_airport_id(df, city_name_encoded):
    """
    Function to search a DataFrame for the corresponding OriginAirportID
    based on either OriginCityNameEncoded or DestCityNameEncoded.

    Parameters:
    df (DataFrame): The DataFrame containing OriginCityNameEncoded, OriginAirportID,
                    DestCityNameEncoded, and DestAirportID columns.
    city_name_encoded (str or int): The value of OriginCityNameEncoded or DestCityNameEncoded to search for.

    Returns:
    int or None: The corresponding OriginAirportID if found, otherwise None.
    """
    try:

        # Search for the corresponding OriginAirportID based on OriginCityNameEncoded
        if 'OriginCityNameEncoded' in df.columns:
            origin_airport_id = df.loc[df['OriginEncoded'] == city_name_encoded, 'OriginAirportID'].iloc[0]
            origin_city_name = df.loc[df['OriginEncoded'] == city_name_encoded, 'OriginCityNameEncoded'].iloc[0]
            return origin_airport_id, origin_city_name
        elif 'DestCityNameEncoded' in df.columns:
            origin_airport_id = df.loc[df['DestEncoded'] == city_name_encoded, 'DestAirportID'].iloc[0]
            origin_city_name = df.loc[df['DestEncoded'] == city_name_encoded, 'DestCityNameEncoded'].iloc[0]

            return origin_airport_id, origin_city_name
        else:
            # If neither column exists, return None
            return None
    except IndexError:
        # If no matching OriginCityNameEncoded or DestCityNameEncoded found, return None
        return None


def generate_response(request):
    if request.method == 'POST':
        try:
            # Extract model and prompt from POST request
            model = "llama2"
            prompt = request.POST.get('prompt')

            # Check if model and prompt are provided
            if model and prompt:
                # Form the data with model, prompt, and stream
                data = {
                    'model': model,
                    'prompt': prompt,
                    'stream': False  # Assuming stream is always False
                }

                headers = {
                    'User-Agent': 'MyPythonClient/1.0',
                    'Accept': '*/*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Content-Type': 'application/json'
                }

                # Make a POST request with the formed JSON data
                response = requests.post('http://localhost:11434/api/generate', json=data, headers=headers)
                response_data = response.json()


                # Check if response content is None
                if response_data is None:
                    return JsonResponse({'error': 'Received empty response'}, status=500)

                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': 'Model and prompt parameters are required'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
