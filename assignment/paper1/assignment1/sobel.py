import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the industrial image
# Example: A sample metallic or machinery image
img = cv2.imread('industrial_sample.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Apply Sobel Edge Detection
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)   # X direction
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)   # Y direction

# Gradient magnitude
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Step 3: Apply Canny Edge Detection
edges_canny = cv2.Canny(img, 100, 200)  # thresholds (lower, upper)

# Step 4: Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobel Edge Detection")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Canny Edge Detection")
plt.imshow(edges_canny, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
