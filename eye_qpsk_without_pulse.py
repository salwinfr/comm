import numpy as np
import matplotlib.pyplot as plt
 -------- Parameters --------
num_symbols = 5000
samples_per_symbol = 10  # Oversampling
snr_values = [-10, -5, 0, 5]
# -------- Generate Random Bits and Map to QPSK Symbols --------
bits = np.random.randint(0, 2, 2 * num_symbols).reshape(-1, 2)
symbol_map = {
    (0, 0): 1 + 1j,
    (0, 1): -1 + 1j,
    (1, 1): -1 - 1j,
    (1, 0): 1 - 1j
}
symbols = np.array([symbol_map[tuple(b)] for b in bits]) / np.sqrt(2)  # Normalized
# -------- Oversample --------
tx_signal = np.repeat(symbols, samples_per_symbol)

# -------- Add AWGN Noise --------
def add_awgn(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# -------- Plot Eye Diagrams --------
fig, axes = plt.subplots(len(snr_values), 2, figsize=(12, 8))
fig.suptitle("QPSK Eye Diagram (No Pulse Shaping)", fontsize=16)

for i, snr in enumerate(snr_values):
    rx_signal = add_awgn(tx_signal, snr)
    I = rx_signal.real
    Q = rx_signal.imag

    # Extract overlapping traces
    num_traces = 100
    seg_len = 2 * samples_per_symbol
    I_samples = I[:num_traces * seg_len].reshape(num_traces, seg_len)
    Q_samples = Q[:num_traces * seg_len].reshape(num_traces, seg_len)

    # Plot I component
    for trace in I_samples:
        axes[i, 0].plot(trace, color='green', alpha=0.3)
    axes[i, 0].set_title(f'I Component | SNR = {snr} dB')
    axes[i, 0].grid(True)

    # Plot Q component
    for trace in Q_samples:
        axes[i, 1].plot(trace, color='purple', alpha=0.3)
    axes[i, 1].set_title(f'Q Component | SNR = {snr} dB')
    axes[i, 1].grid(True)

    for j in range(2):
        axes[i, j].set_xlabel("Samples")
        axes[i, j].set_ylabel("Amplitude")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()








import numpy as np
import matplotlib.pyplot as plt

# ------------------- Parameters -------------------
L = 4                    # Oversampling factor (samples per symbol)
n_samples = 3 * L        # Samples per eye trace (3 symbols per trace)
n_traces = 100           # Number of traces in eye diagram
SNR_values = [-10, -5, 0, 5, 10]  # SNR values (dB)
N_bits = 1000            # Total number of random bits (must be even for QPSK)

# ------------------- Bit Generation -------------------
bits = np.random.randint(0, 2, N_bits)  # Random 0s and 1s

# ------------------- QPSK Modulation -------------------
# Mapping: (I, Q) based on bit pairs
bit_to_symbol = {
    (0, 0): 1 + 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): -1 - 1j
}

# Reshape bits into pairs
bit_pairs = bits.reshape(-1, 2)

# Map each bit pair to corresponding QPSK symbol
symbols = np.array([bit_to_symbol[tuple(pair)] for pair in bit_pairs])

# ------------------- Upsampling -------------------
# Insert L-1 zeros between each symbol (complex array)
signal = np.zeros(len(symbols) * L, dtype=complex)
signal[::L] = symbols

# ------------------- AWGN Noise Function -------------------
def add_awgn(SNR_dB, n):
    N0 = 1 / (10 ** (SNR_dB / 10))  # Convert SNR from dB to linear scale
    noise_real = np.random.normal(0, np.sqrt(N0 / 2), n)
    noise_imag = np.random.normal(0, np.sqrt(N0 / 2), n)
    return noise_real + 1j * noise_imag  # Complex AWGN

# ------------------- Plotting Eye Diagrams -------------------
plt.figure(figsize=(12, 8))  # Set figure size

for idx, SNR in enumerate(SNR_values):
    received = signal + add_awgn(SNR, len(signal))  # Add AWGN noise
    plt.subplot(2, 3, idx + 1)  # Create subplot grid (2 rows x 3 columns)

    for k in range(n_traces):
        start = k * n_samples
        end = start + n_samples
        if end >= len(received):
            break
        segment = received[start:end]
        t = np.linspace(0, len(segment) - 1, len(segment)) / L
        plt.plot(t, segment.real, color='blue', alpha=0.5)  # Plot real part

    plt.title(f"Eye Diagram (SNR = {SNR} dB)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


