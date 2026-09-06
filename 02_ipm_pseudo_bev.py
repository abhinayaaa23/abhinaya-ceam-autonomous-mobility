import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\abhin\Downloads\extracted_frames\frame_1359.png"
OUTPUT_PATH = r"C:\Users\abhin\Downloads\ipm_result.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

# Order: top-left, top-right, bottom-right, bottom-left
src = np.float32([
    [379, 289],
    [557, 288],
    [531, 373],
    [433, 373]
])

width, height = 640, 480
dst = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

H, _ = cv2.findHomography(src, dst)
bev = cv2.warpPerspective(image, H, (width, height))

cv2.imwrite(OUTPUT_PATH, bev)
print("IPM complete. Saved:", OUTPUT_PATH)

cv2.imshow("Original", image)
cv2.imshow("IPM / Pseudo-BEV", bev)
cv2.waitKey(0)
cv2.destroyAllWindows()
