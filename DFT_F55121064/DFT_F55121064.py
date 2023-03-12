#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membuat gambar dalam skala abu-abu
img = cv2.imread('hp.jpg', 0)

# Menenrapkan DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Menghitung bedarnya spektrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Menampilkan spektrum asli dan magnitudo secara berdampingan
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()