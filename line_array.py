# every mic is spaced 6.35 cm apart, track 1 is the leftmost, track 4 the rightmost

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.doa import SRP
from scipy.io import wavfile

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000
nfft = 2048
mic_spacing = 0.0635  # 5 cm
num_mics = 4
signal_duration = 10.0  # seconds

# -----------------------------
# MICROPHONE ARRAY GEOMETRY
# (4 microphones inline along x-axis)
# -----------------------------
mic_positions = np.array([
    [0.0, mic_spacing, 2*mic_spacing, 3*mic_spacing],
    [0.0, 0.0,         0.0,           0.0],
    [0.0, 0.0,         0.0,           0.0]
])

# -----------------------------
# LOAD SIGNALS AND COMBINE IN ONE
# -----------------------------

files = [
    "4mic_array/Tr1.wav",
    "4mic_array/Tr2.wav",
    "4mic_array/Tr3.wav",
    "4mic_array/Tr4.wav"
]

signals = []
sample_rates = []

for f in files:
    fs, audio = wavfile.read(f)
    sample_rates.append(fs)

    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

        # Normalize integer audio
        if audio.max() > 1.0:
            audio /= np.iinfo(np.int16).max

    signals.append(audio)

signals = np.stack(signals, axis=0)


# -----------------------------
# SRP-PHAT DOA ESTIMATION
# -----------------------------
doa = SRP(
    mic_positions,
    fs=fs,
    nfft=nfft,
    num_src=1,
    mode="far"
)

doa.locate_sources(signals)

# Estimated angle
estimated_azimuth = doa.azimuth_recon[0]
estimated_azimuth_deg = np.rad2deg(estimated_azimuth)

print(f"Estimated DOA: {estimated_azimuth_deg:.2f} degrees")

# -----------------------------
# VISUALIZATION (POLAR PLOT)
# -----------------------------
angles = doa.azimuth_grid
power = doa.grid.values

# Normalize power for plotting
power = power / np.max(power)

plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

ax.plot(angles, power, label="SRP-PHAT response")
ax.plot(
    estimated_azimuth,
    1.0,
    'ro',
    label=f"Estimated DOA = {estimated_azimuth_deg:.1f}°"
)

ax.set_theta_zero_location("E")
ax.set_theta_direction(-1)
ax.set_title("SRP-PHAT Direction of Arrival")
ax.legend(loc="upper right")

plt.show()
