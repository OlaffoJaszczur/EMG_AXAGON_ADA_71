# EMG Data Acquisition from Axagon ADA-71 Sound Card

Python tools for acquiring EMG (electromyography) signals from the Axagon ADA-71 USB audio interface with left/right microphone inputs.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your Device
Run this to list available audio devices:
```python
import sounddevice as sd
print(sd.query_devices())
```
Ensure you see your Axagon ADA-71 device with at least 2 input channels.

## Usage

### Basic Recording (10 seconds)
```python
from emg_acquisition import EMGAcquisition

emg = EMGAcquisition(device_name="Axagon", sample_rate=44100, channels=2)
emg.start_recording()  # Runs in background thread
import time
time.sleep(10)
emg.stop_recording()
emg.save_data()
emg.plot_data()
```

### Real-Time Visualization
```bash
python realtime_plot.py
```
Shows live EMG signals from both channels while recording.

## Key Features

- **Automatic Device Detection**: Finds Axagon ADA-71 by name
- **Stereo Input**: Captures both L and R channels simultaneously
- **Real-time Processing**: Non-blocking background recording with threading
- **Data Saving**: Exports as NPZ + JSON metadata
- **Statistics**: RMS, peak-to-peak, mean, std dev per channel
- **Visualization**: Static and real-time plots

## Configuration

### Sample Rate
- **44100 Hz** (default): Standard for audio interfaces
- **48000 Hz**: Professional audio standard
- Choose based on your EMG bandwidth requirements

### EMG Signal Considerations
- Typical EMG amplitude: 10 µV - 5 mV
- Useful bandwidth: 10 Hz - 1 kHz (with anti-aliasing)
- May need amplification/conditioning before sound card input

## File Outputs

- `emg_data_YYYYMMDD_HHMMSS.npz`: Binary data (load with `np.load()`)
- `emg_data_YYYYMMDD_HHMMSS_metadata.json`: Acquisition parameters

## Filtering (Optional)

For EMG, typically apply:
```python
from scipy import signal

# High-pass filter (remove motion artifact)
sos = signal.butter(4, 20, 'hp', fs=44100, output='sos')
data_filtered = signal.sosfilt(sos, data)
```

## Troubleshooting

- **Device not found**: Check `sd.query_devices()`, verify USB connection
- **No signal**: Check microphone input levels in Windows Sound settings
- **Noise**: Ensure proper grounding of EMG sensors and cables
- **Buffer underrun**: Reduce block_size or close other audio apps
