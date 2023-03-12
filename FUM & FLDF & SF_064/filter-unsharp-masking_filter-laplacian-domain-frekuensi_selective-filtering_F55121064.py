#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2
import numpy as np

# Memuat gambar
img = cv2.imread('sudam.jpg')

# Mengkonversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filter Unsharp Masking
blur = cv2.GaussianBlur(gray, (0, 0), 3)
unsharp_mask = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# Filter Laplacian Domain Frekuensi
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Selective Filtering
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
adaptive_threshold = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# Menampilkan semuanya
cv2.imshow('Gambar Asli', img)
cv2.imshow('Unsharp Masking', unsharp_mask)
cv2.imshow('Laplacian Domain Frekuensi', img_back)
cv2.imshow('Selective Filtering', adaptive_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()