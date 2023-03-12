#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np

# Membaca citra
img = cv2.imread('polkadot.jpg')

# Mengkonversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menerapkan max filter dengan kernel 3x3 sebanyak 3 kali
kernel = np.ones((3,3), np.uint8)
max1 = cv2.dilate(gray, kernel, iterations=1)
max2 = cv2.dilate(max1, kernel, iterations=1)
max3 = cv2.dilate(max2, kernel, iterations=1)

# Menampilkan citra asli dan hasil max filter sebanyak 3 kali
cv2.imshow('Citra Asli', gray)
cv2.imshow('(1) Max Filter', max1)
cv2.imshow('(2) Max Filter', max2)
cv2.imshow('(3) Max Filter', max3)

# Tunggu hingga tombol keyboard ditekan
cv2.waitKey(0)

# Tutup semua jendela
cv2.destroyAllWindows()
