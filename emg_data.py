import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import logging

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Define sampling rate (adjust as needed)
emg_sampling_rate = 1000  # in Hz

# Function to Perform EMG Data Analysis
def analyze_emg_data(emg_data):
    # Initialize the scaler
    scaler = StandardScaler()

    # Standardize the EMG signal
    emg_data['EMG_CAL'] = scaler.fit_transform(emg_data[['EMG_CAL']])

    # Display data samples
    st.subheader('Data Sample')
    st.write(emg_data.head())

    # Plot the EMG signal
    st.subheader('EMG Signal')
    st.line_chart(emg_data['EMG_CAL'])

    # EMG Signal Processing and Muscle Activity Analysis
    st.header('EMG Signal Processing and Muscle Activity Analysis')

    # Get the EMG signal
    emg_signal = emg_data['EMG_CAL'].values

    # Clean the EMG signal
    emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=emg_sampling_rate)

    # Compute the envelope of the EMG signal (using RMS)
    emg_envelope = nk.emg_amplitude(emg_cleaned)
    emg_data['EMG_Envelope'] = emg_envelope

    # Plot the EMG envelope
    st.subheader('EMG Envelope (Muscle Activity Level)')
    st.line_chart(emg_data['EMG_Envelope'])

    # Estimate Muscle Activation Level
    st.header('Muscle Activation Level Estimation')

    # Define thresholds for muscle activation
    activation_threshold = emg_data['EMG_Envelope'].mean() + emg_data['EMG_Envelope'].std()

    # Identify periods of high muscle activation
    emg_data['High_Activation'] = emg_data['EMG_Envelope'] > activation_threshold

    # Visualize muscle activation over time
    st.subheader('Estimated Muscle Activation Over Time')
    st.line_chart(emg_data['High_Activation'].astype(int))

    # Display percentage of time under high muscle activation
    high_activation_percentage = emg_data['High_Activation'].mean() * 100
    st.write(f"Percentage of time under high muscle activation: {high_activation_percentage:.2f}%")

    # Explanation of Muscle Activation Levels
    st.subheader('Explanation of Muscle Activation Levels')
    st.write("""
    - The **EMG Envelope** represents the overall muscle activity level by taking the root mean square (RMS) of the EMG signal.
    - The **activation threshold** is calculated as the mean plus one standard deviation of the envelope values. This threshold helps identify periods of significant muscle activity.
    - **High Activation** indicates that the muscle is exerting force above the normal resting level, suggesting periods of physical effort or tension.
    """)

# Streamlit App Title and Description
st.title("EMG Data Analysis for Muscle Activation Estimation")
st.write("""
This dashboard allows you to upload your EMG (Electromyography) data.
After uploading, the data will be analyzed to estimate muscle activation levels.
""")

# File Upload Section
st.header('Upload Your Data')

uploaded_file = st.file_uploader(
    "Choose a CSV file containing your EMG data.",
    type="csv"
)

if uploaded_file is not None:
    try:
        # Read and process the uploaded CSV file
        emg_data = pd.read_csv(
            uploaded_file,
            skiprows=[0, 2],
            names=[
                'Timestamp_Unix_CAL',
                'EMG_CAL'
            ],
            usecols=[0, 1]
        )

        # Data preprocessing steps
        emg_data.dropna(how='all', inplace=True)
        emg_data['Timestamp_Unix_CAL'] = pd.to_numeric(emg_data['Timestamp_Unix_CAL'], errors='coerce')
        emg_data['EMG_CAL'] = pd.to_numeric(emg_data['EMG_CAL'], errors='coerce')
        emg_data.dropna(inplace=True)
        emg_data.reset_index(drop=True, inplace=True)
        emg_data['Timestamp'] = pd.to_datetime(emg_data['Timestamp_Unix_CAL'], unit='ms')
        emg_data.set_index('Timestamp', inplace=True)

        # Analyze the data
        analyze_emg_data(emg_data)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info('Please upload a CSV file to proceed.')
