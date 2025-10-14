import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_tablet_defects(image_path, expected_count):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Preprocessing (thresholding)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    result = img.copy()
    defects = []

    # Ignore background (label 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Draw bounding box
        cv2.rectangle(result, (x,y), (x+w, y+h), (0,255,0), 2)

        # Defect criteria (example: broken if too small, extra if unexpected)
        if area < 500:   # threshold depends on tablet size
            defects.append(("Broken", (x,y,w,h)))
            cv2.putText(result, "Broken", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Step 3: Missing/Extra tablets
    detected_count = num_labels - 1
    if detected_count < expected_count:
        defects.append(("Missing", None))
    elif detected_count > expected_count:
        defects.append(("Extra", None))

    # Show results
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(gray, cmap='gray'); plt.title("Gray Image")
    plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Detected Tablets")
    plt.show()

    print(f"Expected tablets: {expected_count}, Detected: {detected_count}")
    print("Defects found:", defects)


# ---- Example usage ----
detect_tablet_defects('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/tablet.jpg', expected_count=10)
