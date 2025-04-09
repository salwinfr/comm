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

