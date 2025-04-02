import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Given binary bit stream
bit_stream = np.random.randint(0, 2, 1024)  # Example: Random 1024-bit sequence

# Function to perform MPSK modulation
def modulate(bits, M):
    k = int(np.log2(M))
    bit_groups = bits.reshape(-1, k)
    symbols = np.array([int("".join(map(str, b)), 2) for b in bit_groups])
    angles = 2 * np.pi * symbols / M
    return np.cos(angles) + 1j * np.sin(angles)

# Function to add AWGN noise
def add_noise(signal, snr_db, bits_per_symbol):
    snr_linear = 10**(snr_db / 10)
    noise_std = np.sqrt(1 / (2 * bits_per_symbol * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# Function to demodulate MPSK
def demodulate(received_signal, M):
    angles = np.angle(received_signal)
    decoded_symbols = np.round((angles / (2 * np.pi)) * M) % M
    return decoded_symbols.astype(int)

# Theoretical BER and SER formulas
def theoretical_ber(M, snr_db):
    k = np.log2(M)
    return erfc(np.sqrt(k * 10**(snr_db / 10)) / np.sqrt(2)) / k

def theoretical_ser(M, snr_db):
    return 2 * erfc(np.sqrt(2 * 10**(snr_db / 10)) * np.sin(np.pi / M))

# Parameters
M_values = [2, 4]  # BPSK and QPSK
snr_db_values = np.arange(-10, 11, 2)  # SNR values from -10 to 10 in steps of 2

BER_results = {}
SER_results = {}

# Loop through M values
for M in M_values:
    bits_per_symbol = int(np.log2(M))
    
    # Ensure bit stream length is a multiple of bits_per_symbol
    bit_stream = bit_stream[:len(bit_stream) - (len(bit_stream) % bits_per_symbol)]
    
    # Modulate
    transmitted_symbols = modulate(bit_stream, M)
    
    BER_sim = []
    SER_sim = []
    
    # Loop through SNR values
    for snr_db in snr_db_values:
        received_symbols = add_noise(transmitted_symbols, snr_db, bits_per_symbol)
        decoded_symbols = demodulate(received_symbols, M)
        
        # Convert decoded symbols back to bits
        decoded_bits = np.array([list(np.binary_repr(s, width=bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()
        
        # BER & SER Calculation
        bit_errors = np.sum(decoded_bits[:len(bit_stream)] != bit_stream)
        symbol_errors = np.sum(decoded_symbols[:len(transmitted_symbols)] != np.round(np.angle(transmitted_symbols) / (2 * np.pi) * M) % M)
        
        BER_sim.append(bit_errors / len(bit_stream))
        SER_sim.append(symbol_errors / len(transmitted_symbols))
        
        # Plot Constellation Diagram
        plt.figure(figsize=(5, 5))
        plt.scatter(received_symbols.real, received_symbols.imag, s=1)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Constellation (M={M}, SNR={snr_db} dB)')
        plt.grid(True)
        plt.show()
    
    BER_results[M] = BER_sim
    SER_results[M] = SER_sim
    
    # Plot BER
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_db_values, BER_sim, marker='o', linestyle='-', label=f'BER M={M}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title(f'BER for M={M}')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # Plot SER
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_db_values, SER_sim, marker='s', linestyle='-', label=f'SER M={M}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate')
    plt.title(f'SER for M={M}')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
