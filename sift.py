import cv2
import matplotlib.pyplot as plt

# Load product image and logo template
product_img = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/product.jpeg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/logo.png', cv2.IMREAD_GRAYSCALE)

# ---------------------------
# 1. Template Matching
# ---------------------------
res = cv2.matchTemplate(product_img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Draw bounding box
h, w = template.shape
product_tm = product_img.copy()
cv2.rectangle(product_tm, max_loc, (max_loc[0]+w, max_loc[1]+h), 255, 2)

# ---------------------------
# 2. ORB Feature Matching
# ---------------------------
orb = cv2.ORB_create(500)

# Detect and compute descriptors
kp1, des1 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(product_img, None)

# Brute Force Matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
product_orb = cv2.drawMatches(template, kp1, product_img, kp2, matches[:20], None, flags=2)

# ---------------------------
# Show results
# ---------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(product_tm, cmap='gray')
plt.title("Template Matching Detection")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(product_orb)
plt.title("ORB Feature Matching")
plt.axis("off")

plt.show()
