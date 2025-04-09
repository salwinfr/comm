import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 4  # Oversampling factor
n_samples = 3 * L  # Samples per trace in eye diagram
n_traces = 100  # Number of eye diagram traces
SNR_values = [-10, -5, 0, 5]
N_bits = 1000  # Total number of random bits

# Generate random bits
bits = np.random.randint(0, 2, N_bits)

# BPSK Modulation: 0 -> -1, 1 -> +1
symbols = np.array([1 if bit else -1 for bit in bits])

# Upsample (insert L-1 zeros between samples)
signal = np.zeros(len(symbols) * L)
signal[::L] = symbols

# AWGN Noise function
def noise(SNR_dB, n):
    N0 = 1 / (10 ** (SNR_dB / 10))
    return np.random.normal(0, np.sqrt(N0 / 2), n)

# Plot eye diagrams
plt.figure(figsize=(10, 10))  # Create a figure

for idx, SNR in enumerate(SNR_values):
    received = signal + noise(SNR, len(signal))  # Add noise
    plt.subplot(2, 2, idx + 1) 
    for k in range(n_traces):
        start = k * n_samples
        end = start + n_samples + 1
        if end >= len(received):
            break
        segment = received[start:end]
        t = np.linspace(0, len(segment) - 1, len(segment)) / L
        plt.plot(t, segment, color='blue', alpha=0.6)
    plt.title(f"Eye Diagram (SNR = {SNR} dB)")
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
