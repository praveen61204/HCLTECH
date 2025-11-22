import cv2
import matplotlib.pyplot as plt

# Load image (metal surface with defects)
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/automobile/Big Truck/Image_000004.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Global Thresholding ---
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --- Adaptive Thresholding (Mean and Gaussian) ---
adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 25, 10)
adaptive_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 25, 10)

# --- Otsu's Thresholding ---
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Visualization
plt.figure(figsize=(12,8))
plt.subplot(2,3,1); plt.imshow(gray, cmap='gray'); plt.title("Original Gray")
plt.subplot(2,3,2); plt.imshow(global_thresh, cmap='gray'); plt.title("Global Thresholding")
plt.subplot(2,3,3); plt.imshow(adaptive_mean, cmap='gray'); plt.title("Adaptive Mean")
plt.subplot(2,3,4); plt.imshow(adaptive_gauss, cmap='gray'); plt.title("Adaptive Gaussian")
plt.subplot(2,3,5); plt.imshow(otsu_thresh, cmap='gray'); plt.title("Otsu Thresholding")
plt.tight_layout()
plt.show()
