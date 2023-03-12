#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('kamera.jpg', cv2.IMREAD_GRAYSCALE)

# Mengkonversi citra ke float32
f = np.float32(img)

# Menerapkan FFT
f_fft = np.fft.fft2(f)

# Menggeser frekuensi nol ke tengah
f_fft_shift = np.fft.fftshift(f_fft)

# Mengitung magnitudo dan mengkonversi ke dB
magnitude_spectrum = 20*np.log(np.abs(f_fft_shift))

# Menampilkan citra dan spektrum frekuensi
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spektrum frekuensi'), plt.xticks([]), plt.yticks([])
plt.show()
