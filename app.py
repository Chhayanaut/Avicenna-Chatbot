import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import logging
import time

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------
# Global Variables and Setup
# ------------------------------------------

# Define sampling rates (adjust as needed)
ecg_sampling_rate = 512
emg_sampling_rate = 1000
gsr_sampling_rate = 1000
ppg_sampling_rate = 1000

# ------------------------------------------
# ECG Data Loading and Preprocessing
# ------------------------------------------

# Define the file path
ecg_file = 'data/ecg_data.csv'

ecg_columns = [
    'Timestamp_Unix_CAL',
    'ECG_LA_RA_24BIT_CAL',
    'ECGtoHR_LA_RA_CAL'
]

# Read the CSV file
ecg_data = pd.read_csv(
    ecg_file,
    skiprows=[0, 2],
    names=ecg_columns,
    usecols=[0, 1, 2]
)

# Data cleaning
ecg_data.dropna(how='all', inplace=True)
ecg_data = ecg_data.apply(pd.to_numeric, errors='coerce')
ecg_data.dropna(inplace=True)
ecg_data.reset_index(drop=True, inplace=True)

# Convert timestamp and set as index
ecg_data['Timestamp'] = pd.to_datetime(ecg_data['Timestamp_Unix_CAL'], unit='ms')
ecg_data.set_index('Timestamp', inplace=True)

# Display data sample and plot
st.write('**ECG Data Sample:**')
st.write(ecg_data.head())
st.line_chart(ecg_data['ECG_LA_RA_24BIT_CAL'])

# ------------------------------------------
# ECG Signal Processing
# ------------------------------------------

# Get the ECG signal
ecg_signal = ecg_data['ECG_LA_RA_24BIT_CAL']

# Apply bandpass filter
ecg_filtered = nk.signal_filter(
    ecg_signal,
    sampling_rate=ecg_sampling_rate,
    lowcut=0.5,
    highcut=40,
    method='butterworth',
    order=3
)

# Clean the ECG signal
ecg_cleaned = nk.ecg_clean(ecg_filtered, sampling_rate=ecg_sampling_rate)
ecg_cleaned = pd.Series(ecg_cleaned, index=ecg_signal.index)

# Try different R-peak detection methods
methods = ['neurokit', 'pantompkins1985', 'hamilton2002', 'christov2004', 'gamboa2008']
rpeaks_detected = False

for method in methods:
    ecg_peaks = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=ecg_sampling_rate, method=method)
    rpeaks = ecg_peaks['ECG_R_Peaks']
    num_rpeaks = len(rpeaks)

    if num_rpeaks > 0:
        rpeaks_detected = True
        # Plot ECG signal with R-peaks
        fig, ax = plt.subplots()
        ax.plot(ecg_cleaned.index, ecg_cleaned.values, label='ECG Signal')
        ax.scatter(ecg_cleaned.index[rpeaks], ecg_cleaned.values[rpeaks], color='red', label='R-peaks')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        st.pyplot(fig)
        # Compute HRV features
        hrv_time = nk.hrv_time(ecg_peaks, sampling_rate=ecg_sampling_rate, show=False)
        hrv_freq = nk.hrv_frequency(ecg_peaks, sampling_rate=ecg_sampling_rate, show=False)
        try:
            hrv_nonlinear = nk.hrv_nonlinear(ecg_peaks, sampling_rate=ecg_sampling_rate, show=False)
        except ZeroDivisionError:
            st.write("Unable to compute HRV non-linear features due to division by zero error.")
            hrv_nonlinear = pd.DataFrame()
        # Combine HRV features
        hrv_features = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)
        # Display HRV features
        st.write(f'**HRV Features using {method}:**')
        st.write(hrv_features)
        break

if not rpeaks_detected:
    st.write("No R-peaks detected with any method. Unable to compute HRV indices.")
    hrv_features = pd.DataFrame()

# ------------------------------------------
# EMG Data Loading and Preprocessing
# ------------------------------------------

# Define the file path
emg_file = 'data/emg_data.csv'

# Specify the columns to read
emg_columns = [
    'Timestamp_Unix_CAL',
    'EMG_CH1_24BIT_CAL',
    'EMG_CH1_BandPass_Filter_CAL'
]

# Read the CSV file
emg_data = pd.read_csv(
    emg_file,
    skiprows=[0, 2],
    names=emg_columns,
    usecols=[0, 1, 2]
)

# Data cleaning
emg_data.dropna(how='all', inplace=True)
emg_data['Timestamp_Unix_CAL'] = pd.to_numeric(emg_data['Timestamp_Unix_CAL'], errors='coerce')
emg_data['EMG_CH1_24BIT_CAL'] = pd.to_numeric(emg_data['EMG_CH1_24BIT_CAL'], errors='coerce')
emg_data['EMG_CH1_BandPass_Filter_CAL'] = pd.to_numeric(emg_data['EMG_CH1_BandPass_Filter_CAL'], errors='coerce')
emg_data.dropna(inplace=True)
emg_data.reset_index(drop=True, inplace=True)
emg_data['Timestamp'] = pd.to_datetime(emg_data['Timestamp_Unix_CAL'], unit='ms')
emg_data.set_index('Timestamp', inplace=True)

# Standardize the EMG signal
scaler = StandardScaler()
emg_data['EMG_CH1_24BIT_CAL'] = scaler.fit_transform(emg_data[['EMG_CH1_24BIT_CAL']])

# Display the EMG data
st.write('**EMG Data Sample:**')
st.write(emg_data.head())
st.line_chart(emg_data['EMG_CH1_24BIT_CAL'])

# ------------------------------------------
# EMG Signal Processing
# ------------------------------------------

# Extract the EMG signal
emg_signal = emg_data['EMG_CH1_24BIT_CAL'].values

# Apply a band-pass filter
emg_filtered = nk.signal_filter(
    emg_signal,
    sampling_rate=emg_sampling_rate,
    lowcut=20,
    highcut=450,
    method='butterworth',
    order=4
)

# Add the filtered signal to the DataFrame
emg_data['EMG_Filtered'] = emg_filtered

# Rectify the EMG signal
emg_rectified = np.abs(emg_filtered)
emg_data['EMG_Rectified'] = emg_rectified

# Apply a moving average filter
window_size = int(emg_sampling_rate * 0.05)  # 50 ms window
emg_smoothed = nk.signal_smooth(
    emg_rectified,
    method='moving_average',
    size=window_size
)
emg_data['EMG_Smoothed'] = emg_smoothed

# Compute RMS
window_size_rms = int(emg_sampling_rate * 0.1)  # 100 ms window
emg_rms = np.sqrt(np.convolve(
    emg_rectified ** 2,
    np.ones(window_size_rms) / window_size_rms,
    mode='same'
))
emg_data['EMG_RMS'] = emg_rms

# Compute Mean Absolute Value (MAV)
window_size_mav = int(emg_sampling_rate * 0.1)  # 100 ms window
emg_mav = np.convolve(
    emg_rectified,
    np.ones(window_size_mav) / window_size_mav,
    mode='same'
)
emg_data['EMG_MAV'] = emg_mav

# Visualize the EMG signals
st.subheader('Raw EMG Signal')
st.line_chart(emg_data['EMG_CH1_24BIT_CAL'])

st.subheader('Filtered EMG Signal')
st.line_chart(emg_data['EMG_Filtered'])

st.subheader('Rectified EMG Signal')
st.line_chart(emg_data['EMG_Rectified'])

st.subheader('Smoothed EMG Signal')
st.line_chart(emg_data['EMG_Smoothed'])

st.subheader('EMG RMS')
st.line_chart(emg_data['EMG_RMS'])

st.subheader('EMG MAV')
st.line_chart(emg_data['EMG_MAV'])

# Detect periods of high muscle activation
threshold_rms = emg_data['EMG_RMS'].mean() + 2 * emg_data['EMG_RMS'].std()
emg_data['Muscle_Activation'] = emg_data['EMG_RMS'] > threshold_rms

# Visualize Muscle Activation
st.subheader('Muscle Activation Periods')
st.line_chart(emg_data['Muscle_Activation'].astype(int))

# Define a function to extract EMG features
def extract_emg_features(signal):
    features = {}
    # Root Mean Square (RMS)
    features['EMG_RMS'] = np.sqrt(np.mean(signal ** 2))
    # Mean Absolute Value (MAV)
    features['EMG_MAV'] = np.mean(np.abs(signal))
    # Zero Crossing Rate (ZCR)
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['EMG_ZCR'] = len(zero_crossings) / len(signal)
    # Waveform Length (WL)
    features['EMG_WL'] = np.sum(np.abs(np.diff(signal)))
    # Simple Square Integral (SSI)
    features['EMG_SSI'] = np.sum(signal ** 2)
    return features

# Extract features from the smoothed EMG signal
emg_features = extract_emg_features(emg_smoothed)

# Convert to DataFrame
emg_features = pd.DataFrame([emg_features])

# Display EMG features
st.write('**EMG Features:**')
st.write(emg_features)

# ------------------------------------------
# GSR and PPG Data Loading and Preprocessing
# ------------------------------------------

# Define the file path
gsr_ppg_file = 'data/gsr_ppg_data.csv'

# Specify the columns to read
gsr_ppg_columns = [
    'Timestamp_Unix_CAL',
    'GSR_Skin_Conductance_CAL',
    'PPG_A13_CAL',
    'PPGtoHR_CAL'
]

# Read the CSV file
gsr_ppg_data = pd.read_csv(
    gsr_ppg_file,
    skiprows=[0, 2],
    names=gsr_ppg_columns,
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

# Display data samples
st.write('**GSR and PPG Data Sample:**')
st.write(gsr_ppg_data.head())

# Plot the GSR and PPG signals
st.subheader('GSR Signal')
st.line_chart(gsr_ppg_data['GSR_Skin_Conductance_CAL'])

st.subheader('PPG Signal')
st.line_chart(gsr_ppg_data['PPG_A13_CAL'])

# ------------------------------------------
# GSR Signal Processing
# ------------------------------------------

# Get the GSR signal
gsr_signal = gsr_ppg_data['GSR_Skin_Conductance_CAL']

# Process the EDA signal
eda_signals, info = nk.eda_process(gsr_signal, sampling_rate=gsr_sampling_rate)

# Compute interval-related EDA features
gsr_features = nk.eda_intervalrelated(eda_signals)

# Display GSR features
st.write('**GSR Features:**')
st.write(gsr_features)

# ------------------------------------------
# PPG Signal Processing
# ------------------------------------------

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
        st.write('**PPG HRV Features:**')
        st.write(ppg_hrv)
    except Exception as e:
        st.write(f"An error occurred while computing HRV from PPG: {e}")
        ppg_hrv = pd.DataFrame()
else:
    st.write("No peaks detected in the PPG signal. Unable to compute HRV.")
    ppg_hrv = pd.DataFrame()

# ------------------------------------------
# Combine Features for Modeling
# ------------------------------------------

# Ensure all feature DataFrames are not empty
if hrv_features.empty:
    hrv_features = pd.DataFrame(columns=['HRV_Feature1', 'HRV_Feature2'])  # Replace with actual feature names
if emg_features.empty:
    emg_features = pd.DataFrame(columns=['EMG_Feature1', 'EMG_Feature2'])  # Replace with actual feature names
if gsr_features.empty:
    gsr_features = pd.DataFrame(columns=['GSR_Feature1', 'GSR_Feature2'])  # Replace with actual feature names
if ppg_hrv.empty:
    ppg_hrv = pd.DataFrame(columns=['PPG_HRV_Feature1', 'PPG_HRV_Feature2'])  # Replace with actual feature names

# Combine features into a single DataFrame
features = pd.concat([hrv_features, emg_features, gsr_features, ppg_hrv], axis=1)

# Handle any overlapping column names or indices
features.reset_index(drop=True, inplace=True)

# Display the combined features
st.write('**Combined Features:**')
st.write(features)

# ------------------------------------------
#