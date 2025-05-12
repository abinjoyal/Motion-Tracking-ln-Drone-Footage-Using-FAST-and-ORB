import cv2
import matplotlib.pyplot as plt

# Read the input image
image =cv2.imread("C:\\temerario01.webp",cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(image, None)

# Draw keypoints on the image
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Display the result
plt.imshow(output_image, cmap='gray')
plt.title("ORB Keypoints")
plt.show()
