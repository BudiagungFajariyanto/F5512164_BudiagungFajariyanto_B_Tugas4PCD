#F55121064
#Budi agung Fajariyanto
#B-TI

# Import package yag digunakan
import cv2

# Membaca citra
img = cv2.imread('albert.jpg')

# Mengkonversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menerapkan median filter dengan kernel 3x3 sebanyak 3 kali
median1 = cv2.medianBlur(gray, 3)
median2 = cv2.medianBlur(median1, 3)
median3 = cv2.medianBlur(median2, 3)

# Menampilkan citra asli dan hasil median filter sebanyak 3 kali
cv2.imshow('Citra Asli', gray)
cv2.imshow('(1) Median Filter', median1)
cv2.imshow('(2) Median Filter', median2)
cv2.imshow('(3) Median Filter', median3)

cv2.waitKey(0)
cv2.destroyAllWindows()
