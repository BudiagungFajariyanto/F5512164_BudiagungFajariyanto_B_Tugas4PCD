#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Memuat gambar asli
img = cv2.imread('galaksi.jpg', cv2.IMREAD_GRAYSCALE)

# Menghitung ukuran gambar
rows, cols = img.shape

# Menghitung koordinat titik tengah
crow, ccol = int(rows/2), int(cols/2)

# Membuat filter mask Ideal Lowpass
ideal_mask = np.zeros((rows, cols, 2), np.float32)
radius = 50
ideal_mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

# Membuat filter mask Butterworth Lowpass
butterworth_mask = np.zeros((rows, cols, 2), np.float32)
radius = 50
n = 2
for i in range(rows):
    for j in range(cols):
        butterworth_mask[i,j] = 1 / (1 + ((np.sqrt((i-crow)**2 + (j-ccol)**2)) / radius)*(2*n))

# Melakukan transformasi Fourier
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Menghitung hasil konvolusi dengan filter mask
ideal_result = dft_shift * ideal_mask
butterworth_result = dft_shift * butterworth_mask

# Melakukan inverse transformasi Fourier
ideal_img = cv2.idft(np.fft.ifftshift(ideal_result))
ideal_img = cv2.magnitude(ideal_img[:, :, 0], ideal_img[:, :, 1])

butterworth_img = cv2.idft(np.fft.ifftshift(butterworth_result))
butterworth_img = cv2.magnitude(butterworth_img[:, :, 0], butterworth_img[:, :, 1])

# Menampilkan hasil pengolahan citra
plt.subplot(231), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(ideal_mask[:, :, 0], cmap='gray')
plt.title('Ideal Lowpass Filter Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(ideal_img, cmap='gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(butterworth_mask[:, :, 0], cmap='gray')
plt.title('Butterworth Lowpass Filter Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(butterworth_img, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()