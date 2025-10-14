import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/automobile/Big Truck/Image_000002.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# --- Sobel Edge Detection ---
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

# --- Laplacian Edge Detection ---
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# --- Canny Edge Detection ---
canny = cv2.Canny(blur, 50, 150)  # thresholds can be tuned

# Visualization
plt.figure(figsize=(14,6))
plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Original Gray")
plt.subplot(1,4,2); plt.imshow(sobel, cmap='gray'); plt.title("Sobel Edges")
plt.subplot(1,4,3); plt.imshow(laplacian, cmap='gray'); plt.title("Laplacian Edges")
plt.subplot(1,4,4); plt.imshow(canny, cmap='gray'); plt.title("Canny Edges")
plt.show()
