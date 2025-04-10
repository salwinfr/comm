import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#  Generate random binary bit stream
bit_stream = np.random.randint(0, 2, 1024)  # 1024 random bits

#  Function to perform MPSK modulation
def modulate(bits, M):
    k = int(np.log2(M))
    bit_groups = bits.reshape(-1, k)
    symbols = np.array([int("".join(map(str, b)), 2) for b in bit_groups])
    angles = 2 * np.pi * symbols / M
    return np.cos(angles) + 1j * np.sin(angles)

#  Function to add AWGN noise
def add_noise(signal, snr_db, bits_per_symbol):
    snr_linear = 10**(snr_db / 10)
    noise_std = np.sqrt(1 / (2 * bits_per_symbol * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

#  Function to demodulate MPSK
def demodulate(received_signal, M):
    angles = np.angle(received_signal)
    decoded_symbols = np.round((angles / (2 * np.pi)) * M) % M
    return decoded_symbols.astype(int)

#  Theoretical BER and SER (optional usage)
def theoretical_ber(M, snr_db):
    k = np.log2(M)
    return erfc(np.sqrt(k * 10**(snr_db / 10)) / np.sqrt(2)) / k

def theoretical_ser(M, snr_db):
    return 2 * erfc(np.sqrt(2 * 10**(snr_db / 10)) * np.sin(np.pi / M))

#  Parameters
M_values = [2, 4]  # BPSK (M=2) and QPSK (M=4)
snr_db_values = np.arange(-10, 11, 2)  # Full SNR range: -10 to 10 dB (step = 2)
snr_db_constellation = [-10, 0, 10]    # Plot constellation only for these SNRs

BER_results = {}
SER_results = {}

#  Loop over modulation schemes
for M in M_values:
    bits_per_symbol = int(np.log2(M))
    
    # Truncate bits to make it divisible by bits_per_symbol
    trimmed_bits = bit_stream[:len(bit_stream) - (len(bit_stream) % bits_per_symbol)]
    
    transmitted_symbols = modulate(trimmed_bits, M)
    
    BER_sim = []
    SER_sim = []

    #  Loop over SNR values
    for snr_db in snr_db_values:
        received_symbols = add_noise(transmitted_symbols, snr_db, bits_per_symbol)
        decoded_symbols = demodulate(received_symbols, M)

        # Decode symbols → bits
        decoded_bits = np.array(
            [list(np.binary_repr(s, width=bits_per_symbol)) for s in decoded_symbols]
        ).astype(int).flatten()

        # Error calculations
        bit_errors = np.sum(decoded_bits[:len(trimmed_bits)] != trimmed_bits)
        symbol_errors = np.sum(decoded_symbols != demodulate(transmitted_symbols, M))

        BER_sim.append(bit_errors / len(trimmed_bits))
        SER_sim.append(symbol_errors / len(transmitted_symbols))

        #  Plot constellation for selected SNRs only
        if snr_db in snr_db_constellation:
            plt.figure(figsize=(5, 5))
            plt.scatter(received_symbols.real, received_symbols.imag, s=1, alpha=0.5)
            plt.title(f'Constellation (M={M}, SNR={snr_db} dB)')
            plt.xlabel('In-Phase')
            plt.ylabel('Quadrature')
            plt.grid(True)
            plt.axis('equal')
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.show()

    # Save results
    BER_results[M] = BER_sim
    SER_results[M] = SER_sim

    #  Plot BER curve
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_db_values, BER_sim, marker='o', linestyle='-', label=f'M={M}')
    plt.title(f'Bit Error Rate vs SNR for M={M}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()

    #  Plot SER curve
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_db_values, SER_sim, marker='s', linestyle='-', label=f'M={M}')
    plt.title(f'Symbol Error Rate vs SNR for M={M}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()






from numpy import *
from matplotlib.pyplot import *
from scipy.special import erfc

M_values = [2,4,8,16]
bit_error_array = []
theoretical_ber_array = []
def theoretical_ber(snr_db):
    snr_linear = 10**(snr_db/10)
    return 0.5*erfc(sqrt(2*snr_linear))
def modulate(bitstream,M):
    k = int(log2(M))
    bit_group = bitstream.reshape(-1,k)
    symbols = array([int("".join(map(str,b)),2) for b in bit_group])
    angle = 2*pi*symbols/M
    return cos(angle) + 1j*sin(angle)

def add_awgn(signal,snr_db,bits_per_symbol):
    snr_linear = 10**(snr_db/10)
    noise_std = np.sqrt(1 / (2 * bits_per_symbol * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise
snr_db_range = arange(-10,11,5)
def demod(signal,M):
    angles = angle(signal)
    decoded_symbols = np.round( ( angles/(2*pi) )*M )%M
    return decoded_symbols.astype(int)
for M in M_values:
    bits_per_symbol = int(log2(M))
    bitstream = random.randint(0,2,10000)
    bitstream = bitstream[:len(bitstream)-( len(bitstream) % bits_per_symbol )]
    modulated = modulate(bitstream,M)
    
    for snr_db in snr_db_range:
      recieved_signal = add_awgn(modulated,snr_db,int(log2(M)))
      decoded_symbols = demod(recieved_signal,M)
      decoded_bits = array([list(binary_repr(s,width=bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()
      bit_error = sum(decoded_bits[:len(bitstream)] != bitstream)
      ber = bit_error/len(bitstream)
      bit_error_array.append(ber)
      theoretical = theoretical_ber(snr_db)
      t_ber = theoretical/len(bitstream)
      theoretical_ber_array.append(t_ber)
      scatter(recieved_signal.real,recieved_signal.imag,s=1)
      show()
    semilogy(snr_db_range,bit_error_array)
    semilogy(snr_db_range,theoretical_ber_array)
    show()



