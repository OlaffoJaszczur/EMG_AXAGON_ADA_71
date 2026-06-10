"""
Quick-start example - Run this to verify your Axagon ADA-71 setup
"""
from emg_acquisition import EMGAcquisition, ACQUISITION_DEVICE
import time

print("=== EMG Signal Acquisition Quick Start ===\n")

# Step 1: Find and initialize device
# Device is set by ACQUISITION_DEVICE in emg_acquisition.py (index or name substring);
# override here with e.g. device=39 if needed.
try:
    emg = EMGAcquisition(
        device=ACQUISITION_DEVICE,
        sample_rate=None,  # None = device default (48 kHz for the ADA-71)
        channels=2,
        block_size=2048
    )
except ValueError as e:
    print(f"Error: Device not found!\nMake sure Axagon ADA-71 is connected.\n{e}")
    exit(1)

# Step 2: Start recording
print("\nStarting 5-second test recording...")
import threading
record_thread = threading.Thread(target=emg.start_recording)
record_thread.daemon = True
record_thread.start()

# Wait for recording
time.sleep(5)
emg.stop_recording()
record_thread.join(timeout=2)

# Step 3: Display statistics
print("\n")
emg.get_statistics()

# Step 4: Save data
filename = emg.save_data()

# Step 5: Plot
print("\nDisplaying plot...")
emg.plot_data()

print("\nTest complete! Your setup is working.")
