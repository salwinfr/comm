import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def srrc_pulse(Tsym, beta, L, Nsym):
    """Generates a Square-Root Raised Cosine (SRRC) pulse while handling singularities."""
    t = np.arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = np.zeros_like(t)
    
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
        elif abs(t[i]) == Tsym / (4 * beta):
            p[i] = (beta / np.sqrt(2 * Tsym)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
                   4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
            denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
            p[i] = num / denom
    
    return p / np.sqrt(np.sum(p ** 2))


def upsample_and_filter(symbols, pulse, L):
    """Upsamples the symbols and applies the SRRC filter."""
    upsampled = np.zeros(len(symbols) * L)
    upsampled[::L] = symbols    
    return convolve(upsampled, pulse, mode='full')

def add_awgn(signal, snr_db):
    """Adds AWGN noise to the signal based on SNR (dB)."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def downsample_and_demodulate(received_signal, pulse, L, num_bits):
    """Performs matched filtering, downsampling, and demodulation."""
    matched_output = convolve(received_signal, pulse, mode='full')
    delay = (len(pulse) - 1) // 2
    sampled = matched_output[2 * delay + 1::L]
    detected_symbols = np.where(sampled >= 0, 1, -1)
    
    # Ensure the correct number of bits
    return detected_symbols[:num_bits]

def simulate_pulse_shaping(bits):
    """Runs the pulse shaping simulation with a given binary bit sequence."""
    
    symbols = np.where(bits == 0, -1, 1)  # BPSK Mapping

    # Define parameters
    Tsym, beta, L, Nsym = 1, 0.3, 4, 8
    pulse = srrc_pulse(Tsym, beta, L, Nsym)

    # Transmit signal
    transmitted_signal = upsample_and_filter(symbols, pulse, L)

    snr_values = np.arange(-10, 21, 5)  # SNR from -10 dB to 20 dB
    ber_values = []

    for snr in snr_values:
        received_signal = add_awgn(transmitted_signal, snr)
        detected_symbols = downsample_and_demodulate(received_signal, pulse, L, len(bits))
        
        # Ensure the correct number of bits before reshaping
        recovered_bits = (detected_symbols == 1).astype(np.uint8)
        recovered_bits = np.pad(recovered_bits, (0, 8 - len(recovered_bits) % 8), mode='constant')[:len(bits)]

        # Calculate Bit Error Rate (BER)
        errors = np.sum(recovered_bits != bits) 
        ber = errors / len(bits)
        ber_values.append(ber)

    # Plot SNR vs BER Curve
    plt.figure()
    plt.semilogy(snr_values, ber_values, 'o-', label="Simulated BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("SNR vs BER Curve")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


# Example usage with a random bit array
num_bits = 10000
bits = np.random.randint(0, 2, num_bits, dtype=np.uint8)
simulate_pulse_shaping(bits)
