import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def modulate(bits, M):
    K = int(np.log2(M))
    bits_group = bits.reshape(-1, K)
    symbol = np.array([int("".join(map(str, b)), 2) for b in bits_group])
    angles = 2 * np.pi * symbol / M
    return np.cos(angles) + 1j * np.sin(angles)

def add_noise(snr_db, signal):
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return noise + signal

def demodulate(signal, M):
    angles = np.angle(signal)
    decoded_symbol = np.round(angles * M / (2 * np.pi)) % M
    return decoded_symbol.astype(int)

def theoretical_ber(snr_db, M):
    K = int(np.sqrt(M))
    snr_linear = 10 ** (snr_db / 10)
    return erfc(np.sqrt(K * snr_linear) / np.sqrt(2)) / K

def plot_eye_diagram(signal):
    samples_per_symbol = 10  # Assume 10 samples per symbol for visualization
    signal_sampled = np.real(signal)
    num_symbols = len(signal_sampled) // samples_per_symbol
    eye_data = signal_sampled[:num_symbols * samples_per_symbol].reshape(num_symbols, samples_per_symbol)
    for row in eye_data:
        plt.plot(row, color='blue', alpha=0.5)
    plt.title(f"Eye Diagram for snr={snr_db}")
    plt.grid()
    plt.show()

# Main program
bits = np.random.randint(0, 2, 10000, dtype=int)
M_vals = [2]
snr_db_range = np.arange(-10, 11, 10)
for M in M_vals:
    bits_per_symbol = int(np.log2(M))
    bits = bits[:len(bits) - len(bits) % bits_per_symbol]
    ber = []
    ber1 = []
    transmitted = modulate(bits, M)
    for snr_db in snr_db_range:
        received_signal = add_noise(snr_db, transmitted)
        decoded_symbols = demodulate(received_signal, M)
        decoded_bits = np.array([list(np.binary_repr(s, width=bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()
        bit_error = np.sum(decoded_bits[:len(bits)] != bits)
        ber.append(bit_error / len(bits))
        ber1.append(theoretical_ber(snr_db, M))

        plt.scatter(received_signal.real, received_signal.imag, marker='o', s=1)
        plt.title(f"Constellation Diagram for M={M}, SNR={snr_db}")
        plt.grid()
        plt.show()

        # Generate eye diagram for the received signal
        plot_eye_diagram(received_signal)

    plt.semilogy(snr_db_range, ber1, label="Theoretical BER")
    plt.semilogy(snr_db_range, ber, label="Simulated BER")
    plt.title(f"BER vs SNR for M={M}")
    plt.legend()
    plt.grid()
    plt.show()
