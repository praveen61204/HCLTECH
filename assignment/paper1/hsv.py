import cv2
import numpy as np
import matplotlib.pyplot as plt

def classify_fruit(image_path):
    # Load image
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Define HSV ranges ---
    # Unripe (green)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Ripe (yellow to red)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])

    # Create masks
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_ripe = mask_yellow | mask_red

    # Count pixels in each mask
    green_pixels = cv2.countNonZero(mask_green)
    ripe_pixels = cv2.countNonZero(mask_ripe)

    classification = "Ripe" if ripe_pixels > green_pixels else "Unripe"

    # Visualization
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(mask_green, cmap="gray"); plt.title("Green Mask (Unripe)")
    plt.subplot(1,3,3); plt.imshow(mask_ripe, cmap="gray"); plt.title("Ripe Mask (Yellow/Red)")
    plt.show()

    return classification

# Test the function with sample images
samples = ['D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/fruits/images/apple fruit/Image_4.jpg','D:/personal/OneDrive/Documents/HCL/Imageprocessing/Dataset/fruits/images/apple fruit/Image_5.jpg']
for s in samples:
    result = classify_fruit(s)
    print(f"{s} -> {result}")
