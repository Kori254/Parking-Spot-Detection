import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use absolute paths for better reliability
image_path = "/home/kori/Desktop/School/Computer Vision/Cat/parked_slots.jpeg"
empty_lot_path = "/home/kori/Desktop/School/Computer Vision/Cat/Empty_slot.jpeg"

# Load the images
image = cv2.imread(image_path)
empty_lot = cv2.imread(empty_lot_path)

# Check if images are loaded properly
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()
if empty_lot is None:
    print(f"Error: Could not load image at {empty_lot_path}")
    exit()

# Convert the images to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_empty_lot = cv2.cvtColor(empty_lot, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the current image and the empty parking lot image
difference = cv2.absdiff(gray_empty_lot, gray_image)

# Apply a binary threshold to get a binary image
_, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

# Ensure the image is a single-channel binary image
if thresh.ndim == 3:
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

# Predefined parking space coordinates (example, you need to adjust based on your actual parking lot image)
# Example coordinates format: [(top-left corner), (bottom-right corner)]
parking_spaces = [((50, 50), (150, 150)), ((200, 50), (300, 150))]

# Draw bounding boxes on the original image
for (x1, y1), (x2, y2) in parking_spaces:
    roi = thresh[y1:y2, x1:x2]
    white_pixel_count = cv2.countNonZero(roi)
    
    # Define a threshold for determining if a space is empty or occupied
    if white_pixel_count < 500:  # Adjust this threshold based on your image analysis
        status = "Empty"
        color = (0, 255, 0)  # Green for empty
    else:
        status = "Occupied"
        color = (0, 0, 255)  # Red for occupied
    
    # Draw the rectangle and put the status text
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the final image with detections
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Parking Occupancy Detection')
plt.show()
