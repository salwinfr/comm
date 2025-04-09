import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate normally distributed random variable
samples = np.random.normal(loc=0, scale=1, size=10000)

# Plot histogram and Gaussian fit
plt.figure(figsize=(8, 4))
count, bins, _ = plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', label='Histogram')
x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, loc=0, scale=1)
plt.plot(x, pdf, 'r--', label='Gaussian PDF')
plt.title('Histogram of Gaussian Distributed Random Variable')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# PCM encoding and SQNR calculation
def pcm_encode(signal, N, min_val, max_val):
    levels = 2 ** N
    delta = (max_val - min_val) / levels
    q_indices = np.floor((signal - min_val) / delta).astype(int)
    q_indices = np.clip(q_indices, 0, levels - 1)
    quantized = min_val + (q_indices + 0.5) * delta
    binary_encoded = [format(i, f'0{N}b') for i in q_indices]
    return quantized, binary_encoded

def calculate_sqnr(original, quantized):
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - quantized)**2)
    return 10 * np.log10(signal_power / noise_power)

min_val, max_val = -3, 3
sqnr_values = []

print("\nPCM Encoded Output for N = 4:")
for N in range(2, 7):
    quantized, binary_encoded = pcm_encode(samples, N, min_val, max_val)
    sqnr = calculate_sqnr(samples, quantized)
    sqnr_values.append(sqnr)

    if N == 4:
        print("First 20 Encoded Values:")
        print(binary_encoded[:20])

# Plot SQNR vs N
plt.figure(figsize=(6, 4))
plt.plot(range(2, 7), sqnr_values, marker='o', linestyle='--', color='purple')
plt.title("SQNR vs Number of Bits (PCM)")
plt.xlabel("Number of Bits (N)")
plt.ylabel("SQNR (dB)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(2, 7))
plt.show()
