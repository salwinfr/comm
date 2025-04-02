
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r= 57 #roll number
modr = (r % 5) + 1  #mod(r,5) + 1
fs = 32000 #fm=4, nyquist=2*4=8, 4times nyquist=4*8=32
t = np.linspace(0, 1, fs)  # Time vector=time b/w 0,1

# Generate raised sine wave signal
s = modr * (1 + np.cos(8 * np.pi * t)) / 2

# Quantization levels
L= [4, 8, 16, 32, 64]
SQNR_values = []#to store square to quantization noise ratio val for each val

plt.figure(figsize=(12, 6))

for i in L:
    
    # Perform uniform quantization
    minval, maxval = np.min(s), np.max(s)
    stepsize = (maxval - minval) / (i - 1)
    quantizedsignal = np.round((s - minval) / stepsize) * stepsize + minval#Maps signal values into integer levels (0 to L-1)* step_size + min_val:Converts back to original amplitude scale.
    
    # Compute quantization noise and SQNR
    quantizationerror = s - quantizedsignal
    signal_power = np.mean(s ** 2)
    noise_power = np.mean(quantizationerror ** 2)
    SQNR = 10 * np.log10(signal_power / noise_power)#Converts power ratio to decibels (dB)
    SQNR_values.append(SQNR)

    # Plot quantized signals for visualization

    plt.plot(t, quantizedsignal, label=f"L={i}")

plt.plot(t, s, label="Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Quantized Signal for Different L values")
plt.legend()
plt.show()

# Plot SQNR vs N
N_values = np.log2(L)
plt.figure()

plt.plot(N_values, SQNR_values,)
plt.xlabel("Number of Bits (N = log2(L))")
plt.ylabel("SQNR (dB)")
plt.title("Signal-to-Quantization Noise Ratio vs Number of Bits")
plt.grid(True)
plt.show()
# PCM Modulation for L = 32
L_pcm = 32
stepsize_pcm = (max(s) - min(s)) / (L_pcm - 1)
quantized_signal_pcm = np.round((s - min(s)) / stepsize_pcm)

# Binary Encoding
def decimal_to_binary(decimal_values, bits_per_sample):
    """Converts decimal values to binary representation.

    Args:
        decimal_values: Array of decimal values to encode.
        bits_per_sample: Number of bits to use for each sample.

    Returns:
        A 1D numpy array containing the binary sequence.
    """
    binary_sequence = []
    for value in decimal_values:
        binary = bin(int(value))[2:].zfill(bits_per_sample)  # Convert to binary and pad with zeros
        binary_sequence.extend([int(bit) for bit in binary])
    return np.array(binary_sequence)

bits_per_sample_pcm = int(np.log2(L_pcm))  # Number of bits needed
pcm_output = decimal_to_binary(quantized_signal_pcm, bits_per_sample_pcm)

print("PCM Modulated Output (Binary Sequence):")
print(pcm_output)
print("Length of PCM output:",len(pcm_output))
