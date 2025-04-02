import numpy as np
import matplotlib.pyplot as plt

# Generate Square-Root Raised Cosine (SRRC) Pulse
def srrc_pulse(t, beta, Tsym):
    numerator = (np.sin(np.pi * t * (1 - beta) / Tsym) +
                 4 * beta * t / Tsym * np.cos(np.pi * t * (1 + beta) / Tsym))
    denominator = (np.pi * t / Tsym) * (1 - (4 * beta * t / Tsym) ** 2) * np.sqrt(Tsym)

    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    pulse = numerator / denominator

    # Apply L'Hôpital’s Rule at singularities
    pulse[t == 0] = (1 / np.sqrt(Tsym)) * ((1 - beta) + (4 * beta / np.pi))
    t_singular = Tsym / (4 * beta)
    pulse[np.abs(t - t_singular) < 1e-10] = (beta / np.sqrt(2 * Tsym)) * \
                                            ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                             (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))

    return pulse

# Generate Random Bitstream
num_bits = 10000
binary_bits = np.random.randint(0, 2, num_bits)

# BPSK Mapping (0 → +1, 1 → -1)
bpsk_symbols = 1 - 2 * binary_bits

# Parameters
L = 4  # Oversampling factor
Tsym = 1  # Symbol duration
Nsym = 8  # Filter length in symbols
beta_values = [0.2, 0.5, 0.9]  # Roll-off factors
snr_values = np.arange(-10, 20, 2)  # SNR range in dB

# BER Results Storage
ber_results = {}

# Iterate over different β values
for beta in beta_values:
    print(f"Processing for roll-off factor β = {beta}...", flush=True)
    ber_results[beta] = []

    # Create time vector
    t = np.arange(-Nsym/2, Nsym/2 + 1/L, 1/L)
    srrc_filter = srrc_pulse(t, beta, Tsym)

    # Upsample: Insert L-1 zeros
    upsampled_symbols = np.zeros(L * len(bpsk_symbols))
    upsampled_symbols[::L] = bpsk_symbols

    # Pulse shaping (Tx Filter)
    shaped_signal = np.convolve(upsampled_symbols, srrc_filter, mode='same')

    for snr_db in snr_values:
        print(f"  Processing for SNR = {snr_db} dB...", flush=True)

        # Add AWGN Noise
        snr_linear = 10 ** (snr_db / 10)
        noise_variance = 1 / (2 * snr_linear)  # Assume Es = 1
        noise = np.sqrt(noise_variance) * np.random.randn(len(shaped_signal))
        received_signal = shaped_signal + noise  # BPSK is real, so add real noise

        # Matched Filter (Rx Filter)
        matched_filter = srrc_filter[::-1]
        filtered_output = np.convolve(received_signal, matched_filter, mode='same')

        # Compute Total Delay
        delta = (len(srrc_filter) - 1) // 2  # One filter delay
        total_delay = 2 * delta  # Tx and Rx filter delay
        valid_start = int(total_delay)

        # Downsample at every L-th position
        downsampled_symbols = filtered_output[valid_start::L]

        # Decision Device (Threshold at 0)
        detected_bits = (downsampled_symbols < 0).astype(int)

        # Compute BER
        bit_errors = np.sum(detected_bits != binary_bits[:len(detected_bits)])
        ber = bit_errors / len(detected_bits)
        ber_results[beta].append(ber)

        # Plot Eye Diagram for selected cases
        if snr_db in [-10, 0, 10]:
            plt.figure(figsize=(8, 5))
            nTraces = 100
            nSamples = 3 * L
            for i in range(nTraces):
                start = np.random.randint(0, len(downsampled_symbols) - nSamples)
                plt.plot(np.arange(nSamples), downsampled_symbols[start:start + nSamples], 'b', alpha=0.3)
            
            plt.title(f"Eye Diagram (β={beta}, SNR={snr_db} dB)")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

# Plot SNR vs BER Curve
plt.figure(figsize=(8, 6))
for beta, ber in ber_results.items():
    plt.semilogy(snr_values, ber, marker='o', label=f'β = {beta}')

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("SNR vs BER Curve")
plt.legend()
plt.grid(True, which='both')
plt.show()
