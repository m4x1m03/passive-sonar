# %% [markdown]
# Import Audio
# %%
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use("TkAgg")

fs, audio = wav.read("FLFR.WAV")

# Normalize and split channels
audio = audio.astype(np.float32)
left = audio[:, 0]
right = audio[:, 1]

window_ms = 50
window_size = int(fs * window_ms / 1000)
hop_size = window_size // 2   # 50% overlap

# %% [markdown]
# Band Pass
# %%
from scipy.signal import butter, filtfilt

# Bandpass filter (speech/music range)
def bandpass(signal, fs, low=300, high=3000):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

# %% [markdown]
# GCC Path
# %%
import numpy as np

def gcc_phat(sig, refsig, fs, max_tau=None):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-10  # PHAT weighting
    cc = np.fft.irfft(R, n=n)
    max_shift = int(n/2)
    if max_tau:
        max_shift = min(int(fs*max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift/fs

# %% [markdown]
# ITD
# %% [markdown]
# window_ms = 50
# window_size = int(fs * window_ms / 1000)
# hop_size = window_size // 2
# 
# c = 343.0      # speed of sound
# d = 0.02       # mic spacing
# max_tau = 1.5 * d / c
# 
# angles_clean = []
# last_angle = 0.0
# 
# for start in range(0, len(left)-window_size, hop_size):
#     end = start + window_size
#     l_win = bandpass(left[start:end], fs)
#     r_win = bandpass(right[start:end], fs)
# 
#     if np.max(np.abs(l_win)) < 1e-4:  # skip silent windows
#         angles_clean.append(last_angle)
#         continue
# 
#     tau = gcc_phat(l_win, r_win, fs, max_tau=max_tau)
#     value = np.clip(c * tau / d, -1, 1)
#     angle = np.arcsin(value)
#     last_angle = angle
#     angles_clean.append(angle)
# 
# %% [markdown]
# ILD
# %%
angles_clean = []
last_angle = 0.0

window_ms = 50
window_size = int(fs * window_ms / 1000)
hop_size = window_size // 2

for start in range(0, len(left)-window_size, hop_size):
    end = start + window_size
    l_win = bandpass(left[start:end], fs)
    r_win = bandpass(right[start:end], fs)

    # skip silent windows
    if np.max(np.abs(l_win)) < 1e-4 and np.max(np.abs(r_win)) < 1e-4:
        angles_clean.append(last_angle)
        continue

    # Compute short-time RMS (or energy)
    l_energy = np.sqrt(np.mean(l_win**2))
    r_energy = np.sqrt(np.mean(r_win**2))

    # ILD in linear scale
    ild = r_energy - l_energy  # right - left

    # Map ILD to angle [-90°, 90°] using arctangent scaling
    angle = np.arctan2(ild, (l_energy + r_energy))  # normalized
    last_angle = angle
    angles_clean.append(angle)
# %% [markdown]
# Plot visualize
# %%
# %matplotlib notebook

# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, polar=True)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_rticks([])
ax.set_ylim(0, 1)

point, = ax.plot([0], [1], 'ro', markersize=10)

def update(frame):
    point.set_data([angles_clean[frame]], [1])
    return point,

anim = FuncAnimation(
    fig,
    update,
    frames=len(angles_clean),
    interval=50,
    repeat=True,
    blit=False
)

plt.show(block=True)   # ⬅️ BLOCK execution until window closes
