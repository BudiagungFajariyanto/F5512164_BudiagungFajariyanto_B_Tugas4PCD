#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('mikro.jpg', 0)

# Transformasi Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Gaussian Lowpass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D = 30
Gaussian_LP = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        Gaussian_LP[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / (2 * D ** 2))

Gaussian_LP_shift = np.fft.fftshift(Gaussian_LP)
filtered_Gaussian_LP = fshift * Gaussian_LP_shift
img_Gaussian_LP = np.fft.ifft2(filtered_Gaussian_LP)
img_Gaussian_LP = np.abs(img_Gaussian_LP)

# Ideal Highpass Filter
D0 = 30
Ideal_HP = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if D > D0:
            Ideal_HP[i, j] = 1

Ideal_HP_shift = np.fft.fftshift(Ideal_HP)
filtered_Ideal_HP = fshift * Ideal_HP_shift
img_Ideal_HP = np.fft.ifft2(filtered_Ideal_HP)
img_Ideal_HP = np.abs(img_Ideal_HP)

# Menampilkan gambar hasil filter
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Gambar'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_Gaussian_LP, cmap='gray')
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_Ideal_HP, cmap='gray')
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()