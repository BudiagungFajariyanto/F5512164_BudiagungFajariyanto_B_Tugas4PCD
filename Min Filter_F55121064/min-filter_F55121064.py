#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np

# Membaca citra
img = cv2.imread('bg.jpg')

# Mengkonversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menerapkan min filter dengan kernel 3x3 sebanyak 3 kali
kernel = np.ones((3,3), np.uint8)
min1 = cv2.erode(gray, kernel, iterations=1)
min2 = cv2.erode(min1, kernel, iterations=1)
min3 = cv2.erode(min2, kernel, iterations=1)

# Menampilkan citra asli dan hasil min filter sebanyak 3 kali
cv2.imshow('Citra Asli', gray)
cv2.imshow('(1) Min Filter', min1)
cv2.imshow('(2) Min Filter', min2)
cv2.imshow('(3) Min Filter', min3)

cv2.waitKey(0)
cv2.destroyAllWindows()
