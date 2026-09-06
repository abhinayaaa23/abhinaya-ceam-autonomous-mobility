import cv2
import numpy as np
import matplotlib.pyplot as plt

DEPTH_PATH = r"PATH_TO_KITTI_DEPTH_PNG"
DEPTH_SCALE = 256.0

# Replace with the calibration for your KITTI camera if different.
FX, FY = 721.5377, 721.5377
CX, CY = 609.5593, 172.8540

X_MIN, X_MAX = -20.0, 20.0
Z_MIN, Z_MAX = 0.0, 40.0
RESOLUTION = 0.10

depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
if depth_raw is None:
    raise FileNotFoundError(DEPTH_PATH)

depth = depth_raw.astype(np.float32) / DEPTH_SCALE
h, w = depth.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))
valid = np.isfinite(depth) & (depth > 0)

z = depth[valid]
x = (u[valid] - CX) * z / FX
y = (v[valid] - CY) * z / FY

mask = (x >= X_MIN) & (x <= X_MAX) & (z >= Z_MIN) & (z <= Z_MAX)
x, y, z = x[mask], y[mask], z[mask]

nx = int((X_MAX - X_MIN) / RESOLUTION)
nz = int((Z_MAX - Z_MIN) / RESOLUTION)

ix = ((x - X_MIN) / RESOLUTION).astype(np.int32)
iz = ((z - Z_MIN) / RESOLUTION).astype(np.int32)

inside = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
ix, iz, y = ix[inside], iz[inside], y[inside]

density_bev = np.zeros((nz, nx), dtype=np.float32)
np.add.at(density_bev, (iz, ix), 1)
density_vis = np.log1p(density_bev)

height_temp = np.full((nz, nx), -np.inf, dtype=np.float32)
np.maximum.at(height_temp, (iz, ix), y)
height_bev = np.full((nz, nx), np.nan, dtype=np.float32)
height_bev[height_temp != -np.inf] = height_temp[height_temp != -np.inf]

occupancy_bev = (density_bev > 0).astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(density_vis, origin="lower",
    extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], aspect="auto")
axes[0].set_title("Point Density BEV")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Forward Z (m)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(height_bev, origin="lower",
    extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], aspect="auto")
axes[1].set_title("Maximum Height BEV")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Forward Z (m)")
plt.colorbar(im1, ax=axes[1], label="Height Y (m)")

axes[2].imshow(occupancy_bev, origin="lower",
    extent=[X_MIN, X_MAX, Z_MIN, Z_MAX], aspect="auto")
axes[2].set_title("Occupancy BEV")
axes[2].set_xlabel("X (m)")
axes[2].set_ylabel("Forward Z (m)")

plt.tight_layout()
plt.show()
