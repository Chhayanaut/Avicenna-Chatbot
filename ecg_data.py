import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import (StandardScaler)
import os
import datetime
import logging

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Global Variables and Setup

# Directory to store and retrieve data files
DATA_STORAGE_DIR = 'uploaded_data_ecg'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

# Define sampling rate (adjust as needed)
ecg_sampling_rate = 1000  # in Hz

# Function to Perform ECG Data Analysis
def analyze_ecg_data(ecg_data):
    # Initialize the scaler
    scaler = StandardScaler()

    # Standardize the ECG signal
    ecg_data['ECG_CAL'] = scaler.fit_transform(ecg_data[['ECG_CAL']])

    # Display data samples
    st.subheader('Data Sample')
    st.write(ecg_data.head())

    # Plot the ECG signal
    st.subheader('ECG Signal')
    st.line_chart(ecg_data['ECG_CAL'])

    # ECG Signal Processing and Heart Rate Variability (HRV) Analysis
    st.header('ECG Signal Processing and Heart Rate Variability (HRV) Analysis')

    # Get the ECG signal
    ecg_signal = ecg_data['ECG_CAL'].values

    # Clean the ECG signal
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=ecg_sampling_rate)

    # Detect R-peaks
    ecg_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)

    # Extract peak indices
    r_peak_indices = ecg_peaks['ECG_R_Peaks']
    num_peaks = len(r_peak_indices)
    st.write(f"Number of R-peaks detected: {num_peaks}")

    if num_peaks > 0:
        # Plot ECG signal with detected R-peaks
        fig, ax = plt.subplots()
        time_ecg = np.arange(len(ecg_cleaned)) / ecg_sampling_rate
        ax.plot(time_ecg, ecg_cleaned, label='ECG Signal')
        ax.scatter(time_ecg[r_peak_indices], ecg_cleaned[r_peak_indices], color='red', label='R-peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        st.pyplot(fig)

        # Compute HRV indices from ECG
        try:
            ecg_hrv = nk.hrv(ecg_peaks, sampling_rate=ecg_sampling_rate, show=False)
            st.subheader('ECG HRV Features')
            st.write(ecg_hrv)
        except Exception as e:
            st.write(f"An error occurred while computing HRV from ECG: {e}")
            ecg_hrv = pd.DataFrame()
    else:
        st.write("No R-peaks detected in the ECG signal. Unable to compute HRV.")
        ecg_hrv = pd.DataFrame()

    # Estimating Stress Level
    if not ecg_hrv.empty:
        # Extract the SDNN feature (Standard Deviation of NN intervals)
        sdnn = ecg_hrv['HRV_SDNN'].values[0]

        # Define thresholds for stress level estimation
        sdnn_mean = ecg_hrv['HRV_SDNN'].mean()
        sdnn_std = ecg_hrv['HRV_SDNN'].std()
        sdnn_threshold_low = sdnn_mean - sdnn_std
        sdnn_threshold_high = sdnn_mean + sdnn_std

        # Estimate stress level
        if sdnn < sdnn_threshold_low:
            stress_level = 'High Stress'
        elif sdnn > sdnn_threshold_high:
            stress_level = 'Low Stress'
        else:
            stress_level = 'Moderate Stress'

        st.subheader('Estimated Stress Level')
        st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")

        # Explanation of Stress Level and R-peaks
        st.subheader('Explanation of Stress Level and R-peaks')
        st.write("""
        - **R-peaks** are the points in the ECG signal where the R wave is detected, representing the electrical activity associated with the contraction of the heart's ventricles.
        - The **number of R-peaks** detected is used to calculate **Heart Rate Variability (HRV)**, which is an important indicator of the autonomic nervous system activity.
        - **HRV_SDNN** (Standard Deviation of NN intervals) is a measure used to estimate the variation in time between successive heartbeats.
        - A lower SDNN value can indicate **high stress** levels, as it suggests reduced variability in heart rate, which is often associated with stress or fatigue.
        - A higher SDNN value generally indicates **low stress** levels, suggesting healthy autonomic regulation and adaptability.
        - **Moderate Stress** means that the heart rate variability falls within a range that is neither too low nor too high. It indicates a balanced state where the body is experiencing some stress but is not overwhelmed. This can be a normal response to everyday challenges and is typically not a cause for concern unless persistent.
        """)
    else:
        st.write("Unable to estimate stress level due to lack of HRV data.")

# Streamlit App Title and Description
st.title("ECG Data Analysis for Stress Level Estimation")
st.write("""
This dashboard allows you to upload your ECG (Electrocardiogram) data.
After uploading, the data will be analyzed to estimate stress levels.
The uploaded data will be stored locally on your PC for future use.
""")

# File Upload Section
st.header('Upload Your Data')

uploaded_file = st.file_uploader(
    "Choose a CSV file containing your ECG data.",
    type="csv"
)

if uploaded_file is not None:
    try:
        # Read and process the uploaded CSV file
        ecg_data = pd.read_csv(
            uploaded_file,
            skiprows=[0, 2],
            names=[
                'Timestamp_Unix_CAL',
                'ECG_CAL'
            ],
            usecols=[0, 1]
        )

        # Data preprocessing steps
        ecg_data.dropna(how='all', inplace=True)
        ecg_data['Timestamp_Unix_CAL'] = pd.to_numeric(ecg_data['Timestamp_Unix_CAL'], errors='coerce')
        ecg_data['ECG_CAL'] = pd.to_numeric(ecg_data['ECG_CAL'], errors='coerce')
        ecg_data.dropna(inplace=True)
        ecg_data.reset_index(drop=True, inplace=True)
        ecg_data['Timestamp'] = pd.to_datetime(ecg_data['Timestamp_Unix_CAL'], unit='ms')
        ecg_data.set_index('Timestamp', inplace=True)

        # Save the processed data to CSV
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ecg_data_{timestamp_str}.csv"
        file_path = os.path.join(DATA_STORAGE_DIR, filename)
        ecg_data.to_csv(file_path, index=False)
        st.write(f"Data has been saved as `{filename}` in the `{DATA_STORAGE_DIR}` directory on your PC.")

        # Analyze the data
        analyze_ecg_data(ecg_data)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info('Please upload a CSV file to proceed.')

# Re-analyze Stored Data Section
st.header('Re-analyze Stored Data')

# List all CSV files in the data storage directory
stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv')]

if len(stored_files) == 0:
    st.write('No stored data files found.')
else:
    selected_file = st.selectbox('Select a data file to re-analyze:', stored_files)
    if st.button('Load and Analyze Selected File'):
        file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
        try:
            ecg_data = pd.read_csv(file_path)

            # If necessary, convert 'Timestamp' column to datetime and set as index
            if 'Timestamp' in ecg_data.columns:
                ecg_data['Timestamp'] = pd.to_datetime(ecg_data['Timestamp'])
                ecg_data.set_index('Timestamp', inplace=True)

            # Analyze the data
            analyze_ecg_data(ecg_data)
        except Exception as e:
            st.error(f"An error occurred while loading and analyzing the file: {e}")
    if st.button('Delete Selected File'):
        file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
        try:
            os.remove(file_path)
            st.success(f"File `{selected_file}` has been deleted.")
            # Refresh the list of stored files
            stored_files.remove(selected_file)
            # Rerun the app to refresh the file list
            st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred while deleting the file: {e}")
