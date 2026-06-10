import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from emg_acquisition import EMGAcquisition, ACQUISITION_DEVICE

class RealtimeEMGPlotter:
    """Real-time EMG signal visualization."""

    def __init__(self, emg_acquisition, window_size=2000):
        """
        Args:
            emg_acquisition: EMGAcquisition instance
            window_size: Number of samples to display
        """
        self.emg = emg_acquisition
        self.window_size = window_size
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 6))
        self.lines = []
        self.axes_data = [np.zeros(window_size) for _ in range(2)]

        for ch in range(2):
            line, = self.axes[ch].plot([], [], lw=2, color='blue')
            self.lines.append(line)
            self.axes[ch].set_ylim(-0.5, 0.5)  # Adjust based on your sensor range
            self.axes[ch].set_ylabel(f"Channel {['L', 'R'][ch]} (V)")
            self.axes[ch].grid(True, alpha=0.3)

        self.axes[1].set_xlabel("Time (samples)")
        plt.tight_layout()

    def update(self, frame):
        """Update plot with new data."""
        data = self.emg.get_data()
        if data is not None and len(data) > 0:
            # Get latest window_size samples
            for ch in range(2):
                if len(data) >= self.window_size:
                    self.axes_data[ch] = data[-self.window_size:, ch]
                else:
                    self.axes_data[ch][:len(data)] = data[:, ch]

                self.lines[ch].set_data(range(len(self.axes_data[ch])), self.axes_data[ch])
                self.axes[ch].set_xlim(0, self.window_size)

        return self.lines

    def start(self, interval=50):
        """Start real-time plotting."""
        anim = FuncAnimation(self.fig, self.update, interval=interval, blit=True)
        plt.show()


def main_realtime():
    """Example with real-time plotting."""
    try:
        # Device is set by ACQUISITION_DEVICE in emg_acquisition.py
        emg = EMGAcquisition(
            device=ACQUISITION_DEVICE,
            sample_rate=None,  # None = device default (48 kHz for the ADA-71)
            channels=2,
            block_size=2048
        )

        # Start recording in background
        record_thread = threading.Thread(target=emg.start_recording)
        record_thread.daemon = True
        record_thread.start()

        # Start real-time visualization
        plotter = RealtimeEMGPlotter(emg, window_size=8000)
        plotter.start(interval=50)

        emg.stop_recording()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main_realtime()
