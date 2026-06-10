import sys
import ctypes
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from datetime import datetime
from fnmatch import fnmatch
import json

# Input device used for data acquisition. Run `python -m sounddevice` to list devices.
# Either a sounddevice index (int, e.g. 41) or a name pattern (str, case-insensitive,
# '*' wildcards allowed). NOTE: indices shift when audio devices connect/disconnect,
# so a name pattern is more reliable. The Axagon ADA-71 appears in Windows as
# "USB Sound Device". The Microphone jack is MONO (signal on the left/tip
# contact only), so two-channel recording requires the stereo Line In jack.
ACQUISITION_DEVICE = "Line*USB Sound Device*"

# When ACQUISITION_DEVICE is a name pattern, prefer devices from this host API
# (substring of the host API name, e.g. "WASAPI", "DirectSound", "MME").
# Ignored when ACQUISITION_DEVICE is an index. WASAPI has the lowest latency.
ACQUISITION_HOSTAPI = "WASAPI"

class EMGAcquisition:
    def __init__(self, device=ACQUISITION_DEVICE, sample_rate=None, channels=2, block_size=2048):
        """
        Initialize EMG acquisition from Axagon ADA-71 sound card.

        Args:
            device: Device index (int) or name pattern (str) to search for
            sample_rate: Sampling rate in Hz. None = use the device's default
                (WASAPI shared mode only accepts the rate configured in
                Windows sound settings, 48000 Hz for the ADA-71)
            channels: Number of channels (2 for stereo L/R)
            block_size: Samples per block
        """
        self.channels = channels
        self.block_size = block_size
        self.device_id = self._find_device(device)

        if self.device_id is None:
            raise ValueError(f"Device '{device}' not found among the devices listed above.")

        device_info = sd.query_devices(self.device_id)
        self.sample_rate = int(sample_rate or device_info['default_samplerate'])

        self.is_recording = False
        self.data_buffer = []
        self.lock = threading.Lock()
        print(f"Using device [{self.device_id}]: {device_info['name']} @ {self.sample_rate} Hz")

    def _find_device(self, device):
        """Resolve a device index or name pattern to a usable input device index."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        print("Available audio devices:")
        for i, dev in enumerate(devices):
            api = hostapis[dev['hostapi']]['name']
            print(f"  [{i}] {dev['name']} ({api}) - In: {dev['max_input_channels']}, Out: {dev['max_output_channels']}")

        if isinstance(device, int):
            if device < 0 or device >= len(devices):
                return None
            if devices[device]['max_input_channels'] < self.channels:
                print(f"\nDevice [{device}] '{devices[device]['name']}' has only "
                      f"{devices[device]['max_input_channels']} input channel(s), need {self.channels}.")
                return None
            return device

        pattern = f"*{device.lower()}*"
        for i, dev in enumerate(devices):
            if (fnmatch(dev['name'].lower(), pattern)
                    and dev['max_input_channels'] >= self.channels
                    and ACQUISITION_HOSTAPI.lower() in hostapis[dev['hostapi']]['name'].lower()):
                print(f"\nFound matching device at index {i}")
                return i

        return None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio stream status: {status}")

        with self.lock:
            # indata is shape (frames, channels)
            self.data_buffer.append(indata.copy())

    def start_recording(self):
        """Start recording EMG data in background thread."""
        self.is_recording = True
        self.data_buffer = []

        # WASAPI is COM-based and COM must be initialized per thread; sounddevice
        # only does this for the main thread, so opening the stream from a
        # background thread fails with PaErrorCode -9999 without this.
        com_initialized = False
        if sys.platform == 'win32':
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
            com_initialized = hr in (0, 1)  # S_OK or S_FALSE

        try:
            with sd.InputStream(device=self.device_id,
                              samplerate=self.sample_rate,
                              channels=self.channels,
                              blocksize=self.block_size,
                              callback=self.audio_callback):
                print("Recording started. Press Ctrl+C to stop...")
                while self.is_recording:
                    sd.sleep(100)
        except KeyboardInterrupt:
            self.stop_recording()
        finally:
            if com_initialized:
                ctypes.windll.ole32.CoUninitialize()

    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        print("Recording stopped.")

    def get_data(self):
        """Get recorded data as numpy array."""
        with self.lock:
            if not self.data_buffer:
                return None
            data = np.vstack(self.data_buffer)
        return data

    def save_data(self, filename=None):
        """Save recorded data to file."""
        if filename is None:
            filename = f"emg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

        data = self.get_data()
        if data is None:
            print("No data to save")
            return

        metadata = {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'num_samples': len(data),
            'duration_sec': len(data) / self.sample_rate,
            'timestamp': datetime.now().isoformat()
        }

        np.savez(filename, data=data)

        with open(filename.replace('.npz', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Data saved to {filename}")
        print(f"Duration: {metadata['duration_sec']:.2f} seconds")
        return filename

    def plot_data(self):
        """Plot recorded EMG data."""
        data = self.get_data()
        if data is None:
            print("No data to plot")
            return

        time_axis = np.arange(len(data)) / self.sample_rate

        fig, axes = plt.subplots(self.channels, 1, figsize=(12, 6))
        if self.channels == 1:
            axes = [axes]

        channel_names = ['Left (L)', 'Right (R)']

        for ch in range(self.channels):
            axes[ch].plot(time_axis, data[:, ch], linewidth=0.5)
            axes[ch].set_ylabel(f'{channel_names[ch]} (V)')
            axes[ch].set_xlabel('Time (s)')
            axes[ch].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_statistics(self):
        """Compute and display statistics for each channel."""
        data = self.get_data()
        if data is None:
            print("No data available")
            return

        channel_names = ['Left (L)', 'Right (R)']
        print("\nEMG Signal Statistics:")
        print("-" * 50)

        for ch in range(self.channels):
            signal = data[:, ch]
            print(f"\n{channel_names[ch]}:")
            print(f"  Mean:        {np.mean(signal):.6f} V")
            print(f"  Std Dev:     {np.std(signal):.6f} V")
            print(f"  Min:         {np.min(signal):.6f} V")
            print(f"  Max:         {np.max(signal):.6f} V")
            print(f"  Peak-to-peak: {np.max(signal) - np.min(signal):.6f} V")
            print(f"  RMS:         {np.sqrt(np.mean(signal**2)):.6f} V")


def main():
    """Example usage of EMG acquisition."""
    try:
        # Initialize acquisition (device configured via ACQUISITION_DEVICE at top of file)
        emg = EMGAcquisition(
            device=ACQUISITION_DEVICE,
            sample_rate=None,  # None = device default (48 kHz for the ADA-71)
            channels=2,  # Left and Right
            block_size=2048
        )

        # Start recording in background
        record_thread = threading.Thread(target=emg.start_recording)
        record_thread.daemon = True
        record_thread.start()

        # Record for 10 seconds (example)
        import time
        print("\nRecording for 10 seconds...")
        time.sleep(10)
        emg.stop_recording()

        # Print statistics
        emg.get_statistics()

        # Save data
        emg.save_data()

        # Plot data
        emg.plot_data()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
