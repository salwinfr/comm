import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r = 74  # roll number
modr = (r % 5) + 1  # mod(r,5) + 1
fs = 32000  # fm=4, nyquist=2*4=8, 4times nyquist=4*8=32
t = np.linspace(0, 5, fs)  # Time vector=time b/w 0,1

# Generate raised sine wave signal
s = modr * (1 + np.cos(8 * np.pi * t)) / 2

# Function to compute SQNR
def compute_sqnr(signal, L):
    minval, maxval = np.min(signal), np.max(signal)
    stepsize = (maxval - minval) / (L - 1)
    quantized_signal = np.round((signal - minval) / stepsize) * stepsize + minval
    quantization_error = signal - quantized_signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(quantization_error ** 2)
    return 10 * np.log10(signal_power / noise_power), quantized_signal

# Compute SQNR for L=4 and L=8
L_values = [4, 8]
sqnr_results = []
quantized_signals = []

for L in L_values:
    sqnr, quantized_signal = compute_sqnr(s, L)
    sqnr_results.append(sqnr)
    quantized_signals.append(quantized_signal)

# Plot quantized signals for L=4 and L=8
plt.figure(figsize=(12, 6))
for i, L in enumerate(L_values):
    plt.plot(t[:1000], quantized_signals[i][:1000], label=f'L={L}')

plt.plot(t[:1000], s[:1000], label='Original Signal', linestyle='dashed', linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('PCM Quantized Signal for L=4 and L=8')
plt.legend()
plt.show()

# Plot SQNR Difference
plt.figure()
plt.bar(['L=4', 'L=8'], sqnr_results, color=['blue', 'green'])
plt.xlabel('Quantization Levels (L)')
plt.ylabel('SQNR (dB)')
plt.title('SQNR Comparison for L=4 and L=8')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# Display SQNR difference
sqnr_difference = sqnr_results[1] - sqnr_results[0]
print(f'SQNR Difference between L=8 and L=4: {sqnr_difference:.2f} dB')
