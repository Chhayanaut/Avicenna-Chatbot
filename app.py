import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import logging
import os
import datetime

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------
# Global Variables and Setup
# ------------------------------------------

# Define sampling rates (adjust as needed)
gsr_sampling_rate = 1000  # in Hz
ppg_sampling_rate = 1000  # in Hz

# Initialize the scaler
scaler = StandardScaler()

# Directory to store uploaded data
DATA_STORAGE_DIR = 'uploaded_data'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

# ------------------------------------------
# Streamlit App Title and Description
# ------------------------------------------

st.title("GSR and PPG Data Analysis for Cognitive Load and Emotional State")
st.write("""
This dashboard allows you to upload your GSR (Galvanic Skin Response) and PPG (Photoplethysmography) data.
After uploading, the data will be analyzed to estimate cognitive load and emotional state (happy/sad).
The uploaded data will be stored on the server for future use.
""")

# ------------------------------------------
# File Upload Section
# ------------------------------------------

st.header('Upload Your Data')

uploaded_file = st.file_uploader(
    "Choose a CSV file containing your GSR and PPG data.",
    type="csv"
)

if uploaded_file is not None:
    # Read the uploaded CSV file
    try:
        gsr_ppg_data = pd.read_csv(
            uploaded_file,
            skiprows=[0, 2],
            names=[
                'Timestamp_Unix_CAL',
                'GSR_Skin_Conductance_CAL',
                'PPG_A13_CAL',
                'PPGtoHR_CAL'
            ],
            usecols=[0, 1, 2, 3]
        )

        # Data preprocessing steps
        gsr_ppg_data.dropna(how='all', inplace=True)
        gsr_ppg_data['Timestamp_Unix_CAL'] = pd.to_numeric(gsr_ppg_data['Timestamp_Unix_CAL'], errors='coerce')
        gsr_ppg_data['GSR_Skin_Conductance_CAL'] = pd.to_numeric(gsr_ppg_data['GSR_Skin_Conductance_CAL'], errors='coerce')
        gsr_ppg_data['PPG_A13_CAL'] = pd.to_numeric(gsr_ppg_data['PPG_A13_CAL'], errors='coerce')
        gsr_ppg_data['PPGtoHR_CAL'] = pd.to_numeric(gsr_ppg_data['PPGtoHR_CAL'], errors='coerce')
        gsr_ppg_data.dropna(inplace=True)
        gsr_ppg_data.reset_index(drop=True, inplace=True)
        gsr_ppg_data['Timestamp'] = pd.to_datetime(gsr_ppg_data['Timestamp_Unix_CAL'], unit='ms')
        gsr_ppg_data.set_index('Timestamp', inplace=True)

        # Standardize the GSR and PPG signals
        gsr_ppg_data['GSR_Skin_Conductance_CAL'] = scaler.fit_transform(gsr_ppg_data[['GSR_Skin_Conductance_CAL']])
        gsr_ppg_data['PPG_A13_CAL'] = scaler.fit_transform(gsr_ppg_data[['PPG_A13_CAL']])

        # ------------------------------------------
        # Data Analysis and Visualization
        # ------------------------------------------

        st.header('Data Analysis Results')

        # Display data samples
        st.subheader('Data Sample')
        st.write(gsr_ppg_data.head())

        # Plot the GSR and PPG signals
        st.subheader('GSR Signal')
        st.line_chart(gsr_ppg_data['GSR_Skin_Conductance_CAL'])

        st.subheader('PPG Signal')
        st.line_chart(gsr_ppg_data['PPG_A13_CAL'])

        # ------------------------------------------
        # GSR Signal Processing
        # ------------------------------------------

        st.header('GSR Signal Processing and Cognitive Load Estimation')

        # Get the GSR signal
        gsr_signal = gsr_ppg_data['GSR_Skin_Conductance_CAL']

        # Process the EDA signal
        eda_signals, info = nk.eda_process(gsr_signal, sampling_rate=gsr_sampling_rate)

        # Extract the tonic (SCL) and phasic (SCR) components
        gsr_ppg_data['EDA_Tonic'] = eda_signals['EDA_Tonic']
        gsr_ppg_data['EDA_Phasic'] = eda_signals['EDA_Phasic']

        # Plot the tonic and phasic components
        st.subheader('EDA Tonic Component (Skin Conductance Level)')
        st.line_chart(gsr_ppg_data['EDA_Tonic'])

        st.subheader('EDA Phasic Component (Skin Conductance Response)')
        st.line_chart(gsr_ppg_data['EDA_Phasic'])

        # Compute interval-related EDA features
        gsr_features = nk.eda_intervalrelated(eda_signals)

        # Display GSR features
        st.subheader('GSR Features')
        st.write(gsr_features)

        # ------------------------------------------
        # Estimating Cognitive Load from GSR
        # ------------------------------------------

        # Define thresholds based on the tonic component
        scl_mean = gsr_ppg_data['EDA_Tonic'].mean()
        scl_std = gsr_ppg_data['EDA_Tonic'].std()
        scl_threshold = scl_mean + scl_std

        # Identify periods of high cognitive load
        gsr_ppg_data['High_Cognitive_Load'] = gsr_ppg_data['EDA_Tonic'] > scl_threshold

        # Visualize cognitive load over time
        st.subheader('Estimated Cognitive Load Over Time')
        st.line_chart(gsr_ppg_data['High_Cognitive_Load'].astype(int))

        # Display percentage of time under high cognitive load
        high_load_percentage = gsr_ppg_data['High_Cognitive_Load'].mean() * 100
        st.write(f"Percentage of time under high cognitive load: {high_load_percentage:.2f}%")

        # ------------------------------------------
        # PPG Signal Processing
        # ------------------------------------------

        st.header('PPG Signal Processing and Emotional State Estimation')

        # Get the PPG signal
        ppg_signal = gsr_ppg_data['PPG_A13_CAL'].values

        # Clean the PPG signal
        ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=ppg_sampling_rate)

        # Detect PPG peaks
        ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)

        # Extract peak indices
        ppg_peak_indices = ppg_peaks['PPG_Peaks']
        num_peaks = len(ppg_peak_indices)
        st.write(f"Number of PPG peaks detected: {num_peaks}")

        if num_peaks > 0:
            # Plot PPG signal with detected peaks
            fig, ax = plt.subplots()
            time_ppg = np.arange(len(ppg_cleaned)) / ppg_sampling_rate
            ax.plot(time_ppg, ppg_cleaned, label='PPG Signal')
            ax.scatter(time_ppg[ppg_peak_indices], ppg_cleaned[ppg_peak_indices], color='red', label='Peaks')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            st.pyplot(fig)

            # Compute HRV indices from PPG
            try:
                ppg_hrv = nk.hrv(ppg_peaks, sampling_rate=ppg_sampling_rate, show=False)
                st.subheader('PPG HRV Features')
                st.write(ppg_hrv)
            except Exception as e:
                st.write(f"An error occurred while computing HRV from PPG: {e}")
                ppg_hrv = pd.DataFrame()
        else:
            st.write("No peaks detected in the PPG signal. Unable to compute HRV.")
            ppg_hrv = pd.DataFrame()

        # ------------------------------------------
        # Estimating Emotional State from HRV
        # ------------------------------------------

        if not ppg_hrv.empty:
            # Extract the SDNN feature (Standard Deviation of NN intervals)
            sdnn = ppg_hrv['HRV_SDNN'].values[0]

            # Define thresholds for emotional state estimation
            sdnn_mean = ppg_hrv['HRV_SDNN'].mean()
            sdnn_std = ppg_hrv['HRV_SDNN'].std()
            sdnn_threshold_low = sdnn_mean - sdnn_std
            sdnn_threshold_high = sdnn_mean + sdnn_std

            # Estimate emotional state
            if sdnn < sdnn_threshold_low:
                emotional_state = 'Sadness or Stress'
            elif sdnn > sdnn_threshold_high:
                emotional_state = 'Happiness or Relaxation'
            else:
                emotional_state = 'Neutral'

            st.subheader('Estimated Emotional State')
            st.write(f"Based on HRV analysis, the estimated emotional state is: **{emotional_state}**")
        else:
            st.write("Unable to estimate emotional state due to lack of HRV data.")

        # ------------------------------------------
        # Combine Features for Analysis
        # ------------------------------------------

        # Ensure all feature DataFrames are not empty
        if gsr_features.empty:
            gsr_features = pd.DataFrame(columns=['GSR_Feature1', 'GSR_Feature2'])  # Replace with actual feature names
        if ppg_hrv.empty:
            ppg_hrv = pd.DataFrame(columns=['PPG_HRV_Feature1', 'PPG_HRV_Feature2'])  # Replace with actual feature names

        # Combine features into a single DataFrame
        features = pd.concat([gsr_features, ppg_hrv], axis=1)

        # Handle any overlapping column names or indices
        features.reset_index(drop=True, inplace=True)

        # Display the combined features
        st.header('Combined Features for Cognitive Load and Emotional State Analysis')
        st.write(features)

        # ------------------------------------------
        # Save Uploaded Data to Server
        # ------------------------------------------

        st.header('Data Storage')

        # Generate a unique filename based on timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gsr_ppg_data_{timestamp_str}.csv"
        file_path = os.path.join(DATA_STORAGE_DIR, filename)

        # Save the DataFrame to CSV
        gsr_ppg_data.to_csv(file_path)

        st.write(f"Your data has been saved as `{filename}` in the server's `{DATA_STORAGE_DIR}` directory.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info('Please upload a CSV file to proceed.')
