import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/automobile/Big Truck/Image_000001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Smooth the image to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Apply edge detection (Sobel + Laplacian for highlighting)
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

laplacian = cv2.Laplacian(blur, cv2.CV_64F)

# Step 3: Combine results to highlight scratches/dents
edges = cv2.convertScaleAbs(sobel + laplacian)

# Step 4: Threshold to make scratches/dents visible
_, binary = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)

# Show results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Original (Gray)")
plt.subplot(1,3,2); plt.imshow(edges, cmap='gray'); plt.title("Enhanced Edges")
plt.subplot(1,3,3); plt.imshow(binary, cmap='gray'); plt.title("Scratches/Dents")
plt.show()
