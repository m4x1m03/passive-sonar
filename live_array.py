import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from matplotlib.animation import FuncAnimation
from filterpy.kalman import KalmanFilter
from collections import deque
import threading
import pyaudio

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000
nfft = 1024
c = 343.0  # speed of sound
mic_spacing = 0.2  # 20 cm
num_mics = 4
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation (235Hz-2.767kHz range)
valid_algorithms = "SRP"

# Audio parameters
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 48000  # Match fs

# Processing parameters
WINDOW_MS = 100
WINDOW_SIZE = int(RATE * WINDOW_MS / 1000)
HOP_SIZE = WINDOW_SIZE // 2

# -----------------------------
# MICROPHONE ARRAY GEOMETRY
# (4 microphones inline along x-axis)
# -----------------------------
R = pra.linear_2D_array([0, 0], 4, 0, mic_spacing)

# -----------------------------
# Kalman Filter Settings
# -----------------------------
min_stdev_deg = 3.0
max_stdev_deg = 45.0
confidence_ref = 4.0  # Will be updated based on running data
max_angular_vel = 90  # degrees per second
change_dir_time = 0.5  # seconds


class SonarArray:
    def __init__(self):
        self.audio_buffer = deque(maxlen=WINDOW_SIZE * CHANNELS * 2)
        self.angles = deque(maxlen=500)  # Store last 500 angle estimates
        self.filtered_angles = deque(maxlen=500)
        self.confidences = deque(maxlen=100)  # Track confidence for adaptive reference
        self.last_angle = 0.0
        self.lock = threading.Lock()
        self.running = False

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Initialize DOA algorithm
        self.doa = pra.doa.algorithms[valid_algorithms](R, fs, nfft, c=c, max_four=4, azimuth=np.linspace(0, np.pi, 180))

        # Initialize Kalman filter
        dt = HOP_SIZE / fs
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix (constant angular velocity model)
        self.kf.F = np.array([[1, dt], [0, 1]])

        # Measurement function (only observe angle)
        self.kf.H = np.array([[1, 0]])

        # Initial covariance
        self.kf.P *= 10.0

        # Measurement noise (will be updated dynamically)
        expected_angle_stdev = 12  # degrees
        self.kf.R = np.array([[(expected_angle_stdev * np.pi / 180) ** 2]])

        # Process noise
        sigma_alpha = np.radians(max_angular_vel / change_dir_time)
        self.kf.Q = sigma_alpha**2 * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # Initial state (will be set on first valid measurement)
        self.kf.x = np.array([[np.pi / 2], [0.0]])  # Start at 90 degrees
        self.kalman_initialized = False

    def wrap_angle(self, a):
        """Wrap angle to [-π, π]"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def map_to_front_hemisphere(self, az):
        """Map angle to front hemisphere [0, π]"""
        if np.isnan(az):
            return np.nan
        az = self.wrap_angle(az)  # Wrap to [-π, π]
        if az < 0:
            az = -az  # Mirror to positive side
        return az

    def confidence_to_R(self, conf):
        """Map SRP confidence to Kalman measurement covariance"""
        conf = np.clip(conf, 0.5, 10.0)
        stdev_deg = max_stdev_deg * (confidence_ref / conf)
        stdev_deg = np.clip(stdev_deg, min_stdev_deg, max_stdev_deg)
        return np.array([[(np.deg2rad(stdev_deg)) ** 2]])

    def process_audio(self, audio_data):
        """Process audio chunk and estimate DOA using SRP-PHAT"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to float32
        audio_array = audio_array.astype(np.float32) / 32768.0

        # Add to buffer
        with self.lock:
            self.audio_buffer.extend(audio_array)

            # Process when we have enough data
            if len(self.audio_buffer) >= WINDOW_SIZE * CHANNELS:
                # Get latest window
                buffer_array = np.array(list(self.audio_buffer)[-WINDOW_SIZE * CHANNELS :])

                # Reshape to (num_channels, num_samples)
                # PyAudio interleaves: [ch0_s0, ch1_s0, ch2_s0, ch3_s0, ch0_s1, ...]
                signals = np.zeros((CHANNELS, WINDOW_SIZE))
                for ch in range(CHANNELS):
                    signals[ch, :] = buffer_array[ch::CHANNELS]

                # Check if signals are too quiet (skip silent windows)
                max_amplitude = np.max(np.abs(signals))
                if max_amplitude < 0.01:  # Threshold for silence
                    # Keep previous angle
                    if len(self.filtered_angles) > 0:
                        self.filtered_angles.append(self.filtered_angles[-1])
                    return

                # Compute STFT for each channel
                X_chunk = np.array(
                    [
                        pra.transform.stft.analysis(sig, nfft, nfft // 2).T
                        for sig in signals
                    ]
                )

                try:
                    # Run DOA estimation
                    self.doa.locate_sources(X_chunk, freq_bins=freq_bins)

                    # Get confidence
                    grid = self.doa.grid.values
                    peak = grid.max()
                    mu = grid.mean()
                    sigma = grid.std()
                    confidence = (peak - mu) / sigma if sigma > 0 else 0.0

                    # Get azimuth
                    az = self.doa.azimuth_recon[0] if self.doa.azimuth_recon.size > 0 else np.nan

                    # Map to front hemisphere
                    az_front = self.map_to_front_hemisphere(az)

                    # Store raw angle
                    self.angles.append(az_front)
                    self.confidences.append(confidence)

                    # Update adaptive confidence reference
                    if len(self.confidences) >= 20:
                        valid_confs = [c for c in list(self.confidences)[-50:] if np.isfinite(c)]
                        if valid_confs:
                            global confidence_ref
                            confidence_ref = np.percentile(valid_confs, 80)
                            confidence_ref = np.clip(confidence_ref, 2.0, 6.0)

                    # Kalman filter update
                    self.kf.predict()

                    if not np.isnan(az_front):
                        # Initialize Kalman filter on first valid measurement
                        if not self.kalman_initialized:
                            self.kf.x = np.array([[az_front], [0.0]])
                            self.kalman_initialized = True

                        # Update measurement noise based on confidence
                        self.kf.R = self.confidence_to_R(confidence)

                        # Wrapped innovation (handle angle wrapping)
                        innovation = az_front - self.kf.x[0, 0]
                        # Wrap innovation to [-π, π]
                        innovation = self.wrap_angle(innovation)
                        z_wrapped = self.kf.x[0, 0] + innovation

                        self.kf.update(np.array([[z_wrapped]]))
                    else:
                        # No measurement - prediction only with increased uncertainty
                        self.kf.P *= 1.05

                    # Keep angle in [0, π]
                    self.kf.x[0, 0] = self.kf.x[0, 0] % np.pi
                    filtered_angle = self.kf.x[0, 0]

                    self.filtered_angles.append(filtered_angle)
                    self.last_angle = filtered_angle

                except Exception as e:
                    print(f"DOA processing error: {e}")
                    # Keep previous angle
                    if len(self.filtered_angles) > 0:
                        self.filtered_angles.append(self.filtered_angles[-1])

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")

        self.process_audio(in_data)
        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        """Start audio capture stream"""
        self.running = True
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback,
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def stop_stream(self):
        """Stop audio capture stream"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def get_current_angle(self):
        """Get the most recent filtered angle estimate"""
        with self.lock:
            if len(self.filtered_angles) > 0:
                return self.filtered_angles[-1]
            return np.pi / 2  # Default to 90 degrees (front)

    def get_raw_angle(self):
        """Get the most recent raw angle estimate"""
        with self.lock:
            if len(self.angles) > 0:
                return self.angles[-1]
            return np.pi / 2


def main():
    """Main function to run real-time DOA tracking"""
    sonar = SonarArray()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_zero_location("E")  # 0° = +x (right)
    ax.set_theta_direction(1)  # CCW
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rmax(1.0)
    ax.set_rticks([])
    ax.set_title("Real-time DOA Tracking (4-Mic Linear Array)", pad=20, fontsize=14, fontweight='bold')

    doa_line, = ax.plot([], [], 'r-', linewidth=3, label='Filtered DOA')
    raw_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.5, label='Raw DOA')
    confidence_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    sonar.start_stream()

    def update(frame):
        filtered_az = sonar.get_current_angle()
        raw_az = sonar.get_raw_angle()

        # Update filtered DOA line
        if not np.isnan(filtered_az):
            doa_line.set_data([filtered_az, filtered_az], [0, 1.0])
        else:
            doa_line.set_data([], [])

        # Update raw DOA line
        if not np.isnan(raw_az):
            raw_line.set_data([raw_az, raw_az], [0, 0.8])
        else:
            raw_line.set_data([], [])

        # Update confidence text
        with sonar.lock:
            if len(sonar.confidences) > 0:
                recent_conf = list(sonar.confidences)[-1]
                confidence_text.set_text(f'Confidence: {recent_conf:.2f}')

        return doa_line, raw_line, confidence_text

    anim = FuncAnimation(
        fig, update, interval=50, blit=True, cache_frame_data=False
    )

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        sonar.stop_stream()
        plt.close()


if __name__ == "__main__":
    main()