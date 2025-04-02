import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

r = 74  # Mean (Roll number)
sigma2 = 1  # Variance

sigma = np.sqrt(sigma2)  # Standard deviation
n_samples = 10000  # No. of samples

# Generate random variables X and Y
X = np.random.normal(r, sigma, n_samples)
Y = np.random.normal(r, sigma, n_samples)

# a) Histogram and PDF of X and Y
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(X, bins=50, kde=True, stat="density", label="X", ax=axes[0], color='blue')
sns.histplot(Y, bins=50, kde=True, stat="density", label="Y", ax=axes[1], color='red')

# Plot theoretical Gaussian PDFs
x_vals = np.linspace(r - 4*sigma, r + 4*sigma, 100)
pdf_vals = norm.pdf(x_vals, r, sigma)
axes[0].plot(x_vals, pdf_vals, 'k-', label='Gaussian PDF')
axes[1].plot(x_vals, pdf_vals, 'k-', label='Gaussian PDF')
axes[0].set_title("Histogram and PDF of X")
axes[1].set_title("Histogram and PDF of Y")
axes[0].legend()
axes[1].legend()
plt.show()

# Plot CDF of X and Y
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.ecdfplot(X, ax=axes[0], color='blue', label="X")
sns.ecdfplot(Y, ax=axes[1], color='red', label="Y")

axes[0].set_title("CDF of X")
axes[1].set_title("CDF of Y")
axes[0].set_xlabel("Value")
axes[1].set_xlabel("Value")
axes[0].set_ylabel("Cumulative Probability")
axes[1].set_ylabel("Cumulative Probability")
axes[0].legend()
axes[1].legend()
plt.show()

# b) PMF Approximation of X and Y
unique_X, counts_X = np.unique(np.round(X), return_counts=True)
unique_Y, counts_Y = np.unique(np.round(Y), return_counts=True)

plt.figure(figsize=(12, 5)) 
plt.subplot(1, 2, 1)
plt.stem(unique_X, counts_X / n_samples, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title("PMF Approximation of X")
plt.xlabel("Value")
plt.ylabel("Probability")

plt.subplot(1, 2, 2)
plt.stem(unique_Y, counts_Y / n_samples, linefmt='r-', markerfmt='ro', basefmt='b-')
plt.title("PMF Approximation of Y")
plt.xlabel("Value")
plt.ylabel("Probability")

plt.show()

# c) Histogram and PDF of Z
Z = X + Y  # New random variable Z
plt.figure(figsize=(7, 5))
sns.histplot(Z, bins=50, kde=True, stat="density", label="Z", color='purple')

# Theoretical Gaussian PDF of Z
r_Z = 2 * r  # Mean of Z
sigma_Z = np.sqrt(2 * sigma2)  # Standard deviation of Z
x_vals_Z = np.linspace(r_Z - 4*sigma_Z, r_Z + 4*sigma_Z, 100)
pdf_vals_Z = norm.pdf(x_vals_Z, r_Z, sigma_Z)
plt.plot(x_vals_Z, pdf_vals_Z, 'k-', label='Gaussian PDF')

plt.title("Histogram and PDF of Z")
plt.legend()
plt.show()

# Mean and Variance of Z
mean_Z = np.mean(Z)
var_Z = np.var(Z)
print(f"Mean of Z: {mean_Z:.2f}")
print(f"Variance of Z: {var_Z:.2f}")
