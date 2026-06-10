# EMG Data Acquisition from Axagon ADA-71 Sound Card

Python tools for acquiring EMG (electromyography) signals from the Axagon ADA-71 USB audio interface with left/right microphone inputs.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your Device
Run this to list available audio devices:
```bash
python -m sounddevice
```
Ensure you see your Axagon ADA-71 (shows up in Windows as "USB Sound Device") with at least 2 input channels. Note that its **Microphone** jack is mono (signal on the left/tip contact only) — for two-channel recording, use the stereo **Line In** jack.

### 3. Configure the Device
Acquisition device selection lives at the top of [emg_acquisition.py](emg_acquisition.py):

- `ACQUISITION_DEVICE`: a sounddevice index (int) or a name pattern (str, case-insensitive, `*` wildcards allowed). A name pattern is recommended since indices shift when audio devices connect/disconnect. Default: `"Line*USB Sound Device*"`.
- `ACQUISITION_HOSTAPI`: preferred host API substring when matching by name (e.g. `"WASAPI"`, `"DirectSound"`, `"MME"`). WASAPI has the lowest latency.

## Usage

### Quick Start
```bash
python quickstart.py
```
Runs a 5-second test recording using `ACQUISITION_DEVICE`, prints statistics, saves the data, and plots it — useful for verifying your setup.

### Basic Recording
```python
from emg_acquisition import EMGAcquisition, ACQUISITION_DEVICE
import threading, time

emg = EMGAcquisition(device=ACQUISITION_DEVICE, sample_rate=None, channels=2, block_size=2048)

record_thread = threading.Thread(target=emg.start_recording)
record_thread.daemon = True
record_thread.start()

time.sleep(10)
emg.stop_recording()
emg.get_statistics()
emg.save_data()
emg.plot_data()
```
`sample_rate=None` uses the device's default sample rate (48000 Hz for the ADA-71 in WASAPI shared mode).

### Real-Time Visualization
```bash
python realtime_plot.py
```
Shows live EMG signals from both channels while recording.

## Key Features

- **Flexible Device Selection**: Resolve the input device by sounddevice index or by name pattern (with host API preference)
- **Stereo Input**: Captures both L and R channels simultaneously
- **Real-time Processing**: Non-blocking background recording with threading (with WASAPI COM initialization handled automatically on Windows)
- **Data Saving**: Exports as NPZ + JSON metadata
- **Statistics**: Mean, std dev, min, max, peak-to-peak, and RMS per channel
- **Visualization**: Static and real-time plots

## Configuration

### Sample Rate
- `sample_rate=None` (default): Uses the device's default sample rate
- **44100 Hz**: Standard for audio interfaces
- **48000 Hz**: Professional audio standard (default for the ADA-71 under WASAPI)
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
