import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/glass.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Preprocessing
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Edge Detection (Canny)
edges = cv2.Canny(blur, 80, 200)

# 3. Morphological Processing
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# 4. Find Contours (Defects)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:  # filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(output, "Defect", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# Show results
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Defect Localization (Bounding Boxes)")
plt.axis("off")

plt.show()
