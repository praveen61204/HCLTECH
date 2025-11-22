import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load fabric image
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/automobile/Big Truck/Image_000003.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Enhance defects (optional: high-pass or simple threshold)
# Here we use adaptive thresholding to highlight irregularities
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)

# Step 2: Morphological operations
kernel = np.ones((5,5), np.uint8)

# Opening removes small noise (false defects)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Closing fills small holes inside detected defects
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# Step 3: Extract defect regions
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = img.copy()
cv2.drawContours(result, contours, -1, (0,0,255), 2)  # red outline defects

# Display
plt.figure(figsize=(12,8))
plt.subplot(2,3,1); plt.imshow(gray, cmap='gray'); plt.title("Original Gray")
plt.subplot(2,3,2); plt.imshow(thresh, cmap='gray'); plt.title("Thresholded Defects")
plt.subplot(2,3,3); plt.imshow(opened, cmap='gray'); plt.title("After Opening")
plt.subplot(2,3,4); plt.imshow(closed, cmap='gray'); plt.title("After Closing")
plt.subplot(2,3,5); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Defects Isolated")
plt.tight_layout()
plt.show()
