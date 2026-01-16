# every mic is spaced 6.35 cm apart, track 1 is the rightmost
# (looking from microphone array forward), track 4 the leftmost
# I am taking the example code provided in the library's GitHub to start and modifying
# it to work with my type of files and my specific problems.
# The provided function to plot the data was also throwing errors, so I created my own.

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from matplotlib.animation import FuncAnimation
from filterpy.kalman import KalmanFilter

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000
nfft = 1024
c = 343.0  # speed of sound
mic_spacing = 0.0635  # 6.35 cm (not good for low-end frequencies)
num_mics = 4
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation (235Hz-2.767kHz range)
chunk_duration = 0.1  # seconds
chunk_size = int(chunk_duration * fs)
hop_size = chunk_size // 2
valid_algorithms = "SRP"

# -----------------------------
# MICROPHONE ARRAY GEOMETRY
# (4 microphones inline along x-axis)
# -----------------------------
R = pra.linear_2D_array([0,0], 4, 0, mic_spacing)

# -----------------------------
# LOAD SIGNALS AND COMBINE IN ONE
# -----------------------------
files = [
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr1.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr2.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr3.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr4.wav"
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
# Compute the STFT frames needed
# -----------------------------
num_samples = signals.shape[1]
num_chunks = (num_samples - chunk_size) // hop_size + 1
times = []
azimuths = []

doa = pra.doa.algorithms[valid_algorithms](R, fs, nfft, c=c, max_four=4)

for i in range(num_chunks):
    start = i * hop_size
    end = start + chunk_size

    chunk = signals[:, start:end]

    # STFT for this chunk
    X_chunk = np.array(
        [
            pra.transform.stft.analysis(sig, nfft, nfft // 2).T
            for sig in chunk
        ]
    )

    doa.locate_sources(X_chunk, freq_bins=freq_bins)


    if doa.azimuth_recon.size == 0:
        az = np.nan
    else:
        az = doa.azimuth_recon[0]

    azimuths.append(az)
    times.append(start / fs)


# -----------------------------
# Kalman Filter for noise reduction
# -----------------------------
#mostly implemented using wikipedia formulas and library resources
dt = hop_size / fs
kf = KalmanFilter(dim_x=2, dim_z=1)

# State transition matrix (assume constat angular velocity)
# this assumption does not mean we assume one single constant velocity
kf.F = np.array([
    [1, dt],
    [0, 1]
])

# Measurement function (only care about angle since we do not have velocity)
kf.H = np.array([[1, 0]])

# Covariance matrix
kf.P *= 10.0

# Measurement noise (12 degree variance)
# we can use lambda/2*pi*D to calculate the standard deviation of an angle for
# lambda = c/f and D is the mic array aperture (here 19cm)
# So for a frequency of 2.7kHz, we have a stdev of 6 degrees
expected_angle_stdev = 12 # what standard deviation we expect on average
kf.R = np.array([[(expected_angle_stdev * np.pi / 180)**2]])

# Process noise (how much we want to allow changes of direction)
max_angular_vel = 120 # maximum angular speed expected
change_dir_time = 0.3 # how many seconds for the speed to change
sigma_alpha = np.radians(max_angular_vel/change_dir_time)
kf.Q = sigma_alpha**2 * np.array([
    [dt**4/4, dt**3/2],
    [dt**3/2, dt**2]
])

#need to wrap the measurements to not have crazy difference between measurements
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

#initial state is set using the first azimuth computed
for az in azimuths:
    if not np.isnan(az):
        #initialization matrix
        kf.x = np.array([[az], [0.0]])
        break

filtered_azimuths = []

for az in azimuths:
    kf.predict()
    if not np.isnan(az):
        # Innovation with angle wrapping
        z = np.array([[az]])
        y = wrap_angle(z[0, 0] - kf.x[0, 0])
        kf.update(np.array([[kf.x[0,0] + y]]))

    kf.x[0,0] = wrap_angle(kf.x[0,0])
    filtered_azimuths.append(kf.x[0,0])

filtered_azimuths = np.array(filtered_azimuths)


# -----------------------------
# Get all angles to the front (invert negative ones)
# -----------------------------
def forward_azimuth(az):
    if np.isnan(az):
        return np.nan
    elif(az) < 0 or az > np.pi:
        az = np.atleast_1d(az)
        return az-np.pi
    else:
        return az

# -----------------------------
# Plot DOA from SRP and animated
# -----------------------------
azimuths = np.array(azimuths)
times = np.array(times)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# Initial DOA line
(doa_line,) = ax.plot([0, 0], [0, 1.0], "r-", linewidth=2)

# Time text
time_text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes, color="black"
)

ax.set_rmax(1.0)
ax.set_rticks([])
ax.set_theta_zero_location("E")   # 0° = +x
ax.set_theta_direction(1)         # CCW
ax.set_thetamin(0)
ax.set_thetamax(180)
ax.set_title("DOA Tracking (0.1 s chunks)")

def update(frame):
    az = forward_azimuth(filtered_azimuths[frame])

    if np.isnan(az):
        doa_line.set_data([], [])
    else:
        doa_line.set_data([az, az], [0, 1.0])

    time_text.set_text(f"t = {times[frame]:.2f} s")

    return doa_line, time_text

interval_ms = int(chunk_duration * 1000 * hop_size / chunk_size)

ani = FuncAnimation(
    fig,
    update,
    frames=len(azimuths),
    interval=interval_ms,
    blit=True,
    repeat=False,
)

plt.show()