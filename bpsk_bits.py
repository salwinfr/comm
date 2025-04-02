import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc  # For theoretical BER calculation

# Generate random binary data (0s and 1s)
num_bits = 10000  # Large bitstream for accurate BER calculation
binary_bits = np.random.randint(0, 2, num_bits, dtype=np.int8)

# BPSK Mapping: 0 → +1, 1 → -1
bpsk_symbols = 1 - 2 * binary_bits  # Purely real values

# Define SNR range
snr_db_range = np.arange(-10, 11, 1)  # SNR from -10 dB to 10 dB
ber_simulated = []  # Store simulated BER
ber_theoretical = []  # Store theoretical BER

# Select SNR values for constellation plots
snr_db_constellation = [-10, 0, 10]  # Low, medium, and high SNR

# Plot constellation diagram for selected SNR values
plt.figure(figsize=(12, 4))

for i, snr_db in enumerate(snr_db_constellation):
    snr_linear = 10**(snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = 1 / snr_linear  # Assume signal power = 1

    # Generate complex Gaussian noise (real + imaginary)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_bits) + 1j * np.random.randn(num_bits))

    # Received noisy symbols (add noise)
    received_signal = bpsk_symbols + noise.real  # Keep only real part for BPSK

    # Scatter plot of received symbols
    plt.subplot(1, 3, i + 1)
    plt.scatter(received_signal, noise.imag, s=1, alpha=0.5, color="b")
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-1, 1)
    plt.title(f"Constellation (SNR={snr_db} dB)")
    plt.xlabel("In-Phase Component")
    plt.ylabel("Quadrature Component")
    plt.grid(True)


    # Bit Error Rate (BER) Calculation
for snr_db in snr_db_range:
    snr_linear = 10**(snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = 1 / snr_linear  # Assume signal power = 1

    # Generate noise
    noise = np.sqrt(noise_power) * np.random.randn(len(bpsk_symbols))

    # Received signal
    received_signal = bpsk_symbols + noise

    # Decision rule: Nearest point mapping
    decoded_symbols = np.where(received_signal >= 0, +1, -1)

    # Convert decoded BPSK symbols to binary bits
    binary_bits_decoded = np.where(decoded_symbols == +1, 0, 1)

    # Count bit errors
    num_bit_errors = np.sum(binary_bits_decoded != binary_bits)
    ber = num_bit_errors / len(binary_bits)
    ber_simulated.append(ber)

    # Calculate theoretical BER for BPSK
    ber_theory = 0.5 * erfc(np.sqrt(snr_linear))
    ber_theoretical.append(ber_theory)

    print(f"SNR (dB): {snr_db}, Simulated BER: {ber:.6f}, Theoretical BER: {ber_theory:.6f}")

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_simulated, 'o-', label="Simulated BER")
plt.semilogy(snr_db_range, ber_theoretical, 's--', label="Theoretical BER")
plt.title("BER vs SNR for BPSK Modulation")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

plt.tight_layout()
plt.show()
