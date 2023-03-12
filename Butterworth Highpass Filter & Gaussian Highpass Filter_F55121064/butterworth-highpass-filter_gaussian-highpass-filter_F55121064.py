import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('kamera.jpg', 0)

# Butterworth Highpass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D = 50
n = 2

# Calculate distance from center of frequency domain
distance = np.sqrt((np.arange(rows)[:, np.newaxis] - crow)**2 + (np.arange(cols)[np.newaxis, :] - ccol)**2)

# Calculate butterworth highpass filter
butterworth_highpass = 1 / (1 + (D / np.where(distance == 0, 1e-6, distance))**(2*n))

# Gaussian Highpass Filter
D0 = 50

# Calculate gaussian highpass filter
gaussian_highpass = 1 - np.exp(-((distance ** 2) / (2 * (D0 ** 2))))

# Apply filters to image
butterworth_result = np.fft.ifft2(np.fft.fftshift(butterworth_highpass) * np.fft.fft2(img)).real
gaussian_result = np.fft.ifft2(np.fft.fftshift(gaussian_highpass) * np.fft.fft2(img)).real

# Display images
cv2.imshow('Gambar Asli', img)
cv2.imshow('Unsharp Masking', butterworth_result)
cv2.imshow('Laplacian Domain Frekuensi', gaussian_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
