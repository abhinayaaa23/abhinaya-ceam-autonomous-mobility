import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\abhin\Downloads\extracted_frames\frame_1359.png"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

src = np.float32([
    [379, 289],
    [557, 288],
    [531, 373],
    [433, 373]
])

grid_image = image.copy()

for i in range(4):
    p1 = tuple(src[i].astype(int))
    p2 = tuple(src[(i + 1) % 4].astype(int))
    cv2.line(grid_image, p1, p2, (0, 0, 255), 2)

for i, point in enumerate(src):
    x, y = point.astype(int)
    cv2.circle(grid_image, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(grid_image, str(i + 1), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

for x in np.linspace(src[0][0], src[1][0], 8):
    t = (x - src[0][0]) / (src[1][0] - src[0][0])
    top = src[0] * (1 - t) + src[1] * t
    bottom = src[3] * (1 - t) + src[2] * t
    cv2.line(grid_image, tuple(top.astype(int)),
             tuple(bottom.astype(int)), (255, 0, 0), 1)

for t in np.linspace(0, 1, 8):
    left = src[0] * (1 - t) + src[3] * t
    right = src[1] * (1 - t) + src[2] * t
    cv2.line(grid_image, tuple(left.astype(int)),
             tuple(right.astype(int)), (255, 0, 0), 1)

width, height = 640, 480
dst = np.float32([
    [0, 0], [width - 1, 0],
    [width - 1, height - 1], [0, height - 1]
])
H, _ = cv2.findHomography(src, dst)
bev = cv2.warpPerspective(grid_image, H, (width, height))

cv2.imwrite(r"C:\Users\abhin\Downloads\ipm_grid_original.png", grid_image)
cv2.imwrite(r"C:\Users\abhin\Downloads\ipm_grid_bev.png", bev)

cv2.imshow("Original + Perspective Grid", grid_image)
cv2.imshow("IPM / Top Down", bev)
cv2.waitKey(0)
cv2.destroyAllWindows()
