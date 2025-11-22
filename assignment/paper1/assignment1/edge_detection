import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the PCB image (replace with your image path)
image = cv2.imread('01_missing_hole_01.jpg', cv2.IMREAD_GRAYSCALE)

# Edge detection (Sobel)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

# Morphology to detect breaks/missing
kernel = np.ones((5, 5), np.uint8)
opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)  # Remove noise
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # Connect close edges

# Find contours for broken tracks/missing points
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    if 20 < cv2.contourArea(contour) < 200:  # Size for defects
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)  # Red annotation

# Display annotated output
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Edges + Morphology')
plt.imshow(closed, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Annotated Defects')
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()
