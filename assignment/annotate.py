import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load PCB image
img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/download.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Preprocessing
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Edge Detection
edges = cv2.Canny(blur, 80, 200)

# 3. Morphological Closing to fill small gaps
kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# 4. Defect detection = difference between closed and original edges
defects = cv2.absdiff(closed, edges)

# Threshold to clean defect map
_, defects_bin = cv2.threshold(defects, 30, 255, cv2.THRESH_BINARY)

# 5. Find contours of defect regions
contours, _ = cv2.findContours(defects_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:  # ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(output, "Defect", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# 6. Display results
plt.figure(figsize=(14,6))

plt.subplot(1,3,1)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(defects_bin, cmap="gray")
plt.title("Detected Defects (Binary Map)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Annotated PCB with Defects")
plt.axis("off")

plt.show()
