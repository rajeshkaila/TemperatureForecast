# Install necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Stations, Daily
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st

# Function to fetch weather data for a given station
def fetch_weather_data(latitude, longitude):
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()

    # Fetch nearby stations
    stations = Stations()
    nearby_stations = stations.nearby(latitude, longitude)

    
    station_id = nearby_stations.fetch(1).index[0]  # Get the station ID from the index
    city_name = nearby_stations.fetch(1)['name']  # Extract city name

    # Fetch daily weather data for the station
    data = Daily(station_id, start_date, end_date).fetch()

    return data, city_name

def lstm_forecast(data, sequence_length=15, epochs=5):
    if data is None or len(data) < sequence_length + 1:
        return None, None

    df = data['tavg']

    generator = TimeseriesGenerator(df, df, length=sequence_length + 1, batch_size=1)

    model = Sequential()
    model.add(LSTM(100, activation="relu", input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['RootMeanSquaredError'])

    model.fit(generator, epochs=epochs)

    # Predictions
    y_pred = model.predict(generator)
    y_pred = y_pred.flatten()

    # Trim the actual values to match the length of predictions
    df_trimmed = df.iloc[-len(y_pred):]

    return df_trimmed, y_pred

def main():
    st.title("Weather Forecast App")

    # Input for latitude and longitude
    latitude = st.number_input("Enter Latitude:")
    longitude = st.number_input("Enter Longitude:")

    # Fetch weather data
    data, city_name = fetch_weather_data(latitude, longitude)

    if data is not None:
        # Display fetched data
        st.subheader(f"Fetched Weather Data for {city_name}:")
        st.write(data.tail())

        # Plot temperature data
        st.subheader("Temperature Data Plot:")
        st.line_chart(data[['tavg', 'tmin', 'tmax']])

        # LSTM Forecast
        df_trimmed, y_pred = lstm_forecast(data)

        if df_trimmed is not None and y_pred is not None:
            # Plot LSTM predictions
            st.subheader("LSTM Forecast Plot:")
            st.line_chart(pd.DataFrame({'Actual': df_trimmed, 'Predicted': y_pred}))

            # Display Root Mean Squared Error
            st.subheader("Root Mean Squared Error:")
            rmse = np.sqrt(np.mean(np.square(df_trimmed - y_pred.flatten())))
            st.write(f"RMSE: {rmse}")

if __name__ == "__main__":
    main()
