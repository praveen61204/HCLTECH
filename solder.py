import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_solder_joints(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Preprocessing
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    result = img.copy()
    good, defective = 0, 0
    defect_types = {"small":0, "large":0, "irregular":0}

    # Step 3: Loop through each solder joint candidate
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Draw bounding box
        cv2.rectangle(result, (x,y), (x+w, y+h), (0,255,0), 2)

        # Feature checks
        circularity = (4 * np.pi * area) / (w*h if w*h > 0 else 1)

        if area < 100:  # too small → missing solder
            defective += 1
            defect_types["small"] += 1
            cv2.putText(result, "Missing", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        elif area > 1000:  # too large → excess solder
            defective += 1
            defect_types["large"] += 1
            cv2.putText(result, "Excess", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        elif circularity < 0.5:  # irregular shape
            defective += 1
            defect_types["irregular"] += 1
            cv2.putText(result, "Irregular", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        else:
            good += 1
            cv2.putText(result, "Good", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    # Step 4: Display results
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(gray, cmap='gray'); plt.title("Gray Image")
    plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Detected Solder Joints")
    plt.show()

    # Step 5: Statistics
    total = good + defective
    print(f"Total solder joints detected: {total}")
    print(f"Good joints: {good}")
    print(f"Defective joints: {defective}")
    print("Defect breakdown:", defect_types)

# ---- Example usage ----
analyze_solder_joints('D:/personal/OneDrive/Documents/HCL/Imageprocessing/Unit 2/Color models/Images/download.jpeg')