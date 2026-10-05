import os

import cv2
import numpy as np
import rosbag2_py

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


# ============================================================
# Configuration
# ============================================================

BAG_PATH = (
    r"C:\Users\abhin\Downloads\test\test4_20260902_134059"
    r"\test4_20260902_134059_0.db3"
)

RGB_TOPIC = "/zedx/zed_node/left/color/rect/image"

OUTPUT_DIR = r"results\bev"

ORIGINAL_PATH = os.path.join(
    OUTPUT_DIR,
    "test4_rgb_original.png"
)

IPM_PATH = os.path.join(
    OUTPUT_DIR,
    "test4_rgb_ipm.png"
)


# ============================================================
# IPM parameters
# ============================================================

# Source points in the ORIGINAL 1920 x 1200 image.
#
# Order:
#   top-left
#   top-right
#   bottom-right
#   bottom-left

SOURCE_POINTS = np.float32([
    [650, 680],
    [1270, 680],
    [1650, 1000],
    [270, 1000]
])


# Size of the generated pseudo-BEV image
OUTPUT_WIDTH = 800
OUTPUT_HEIGHT = 600


# Destination rectangle
DESTINATION_POINTS = np.float32([
    [0, 0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0, OUTPUT_HEIGHT - 1]
])


# ============================================================
# ROS image conversion
# ============================================================

def ros_image_to_bgr(msg):

    if msg.encoding == "rgb8":

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(
            msg.height,
            msg.width,
            3
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    elif msg.encoding == "bgr8":

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(
            msg.height,
            msg.width,
            3
        )

    elif msg.encoding == "bgra8":

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(
            msg.height,
            msg.width,
            4
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    elif msg.encoding == "rgba8":

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(
            msg.height,
            msg.width,
            4
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR
        )

    else:

        raise RuntimeError(
            f"Unsupported image encoding: {msg.encoding}"
        )

    return image


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open ROS 2 bag
    # --------------------------------------------------------

    storage_options = rosbag2_py.StorageOptions(
        uri=BAG_PATH,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()

    print("Opening bag...")

    reader.open(
        storage_options,
        converter_options
    )

    print("Bag opened successfully.")

    # --------------------------------------------------------
    # Find first RGB frame
    # --------------------------------------------------------

    while reader.has_next():

        topic, data, timestamp = reader.read_next()

        if topic != RGB_TOPIC:
            continue

        msg = deserialize_message(
            data,
            Image
        )

        print()
        print("RGB frame found")
        print(f"Timestamp : {timestamp}")
        print(f"Resolution: {msg.width} x {msg.height}")
        print(f"Encoding  : {msg.encoding}")

        image = ros_image_to_bgr(msg)

        break

    else:

        raise RuntimeError(
            f"No image found on topic: {RGB_TOPIC}"
        )

    # --------------------------------------------------------
    # Save original frame
    # --------------------------------------------------------

    cv2.imwrite(
        ORIGINAL_PATH,
        image
    )

    print()
    print("Original frame:")
    print(os.path.abspath(ORIGINAL_PATH))

    # --------------------------------------------------------
    # Draw source quadrilateral for debugging
    # --------------------------------------------------------

    debug_image = image.copy()

    points = SOURCE_POINTS.astype(np.int32)

    cv2.polylines(
        debug_image,
        [points],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4
    )

    for i, point in enumerate(points):

        x, y = point

        cv2.circle(
            debug_image,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug_image,
            str(i + 1),
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Compute homography
    # --------------------------------------------------------

    homography_matrix = cv2.getPerspectiveTransform(
        SOURCE_POINTS,
        DESTINATION_POINTS
    )

    # --------------------------------------------------------
    # Apply IPM
    # --------------------------------------------------------

    ipm_image = cv2.warpPerspective(
        image,
        homography_matrix,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    # --------------------------------------------------------
    # Save IPM result
    # --------------------------------------------------------

    cv2.imwrite(
        IPM_PATH,
        ipm_image
    )

    print()
    print("IPM result:")
    print(os.path.abspath(IPM_PATH))

    # --------------------------------------------------------
    # Display both
    # --------------------------------------------------------

    cv2.imshow(
        "Source Points",
        cv2.resize(
            debug_image,
            (960, 600)
        )
    )

    cv2.imshow(
        "RGB IPM Pseudo-BEV",
        ipm_image
    )

    print()
    print("Press any key in an image window to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()