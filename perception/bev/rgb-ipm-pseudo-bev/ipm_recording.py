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

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "test4_rgb_ipm_recording.mp4"
)


# ============================================================
# IPM parameters
# ============================================================

# Current frozen IPM points
# Order:
# top-left
# top-right
# bottom-right
# bottom-left

SOURCE_POINTS = np.float32([
    [650, 680],
    [1270, 680],
    [1650, 1000],
    [270, 1000]
])

OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 600

DESTINATION_POINTS = np.float32([
    [0, 0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0, OUTPUT_HEIGHT - 1]
])


# ============================================================
# ROS Image → OpenCV BGR
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
    # Calculate homography ONCE
    # --------------------------------------------------------

    homography = cv2.getPerspectiveTransform(
        SOURCE_POINTS,
        DESTINATION_POINTS
    )

    print("Homography calculated.")

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    # ZED test4 RGB stream is approximately 4 FPS
    FPS = 4.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        FPS,
        (
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not open video writer."
        )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_count = 0

    print()
    print("Processing RGB frames...")
    print("Press Q in the preview window to stop early.")
    print()

    while reader.has_next():

        topic, data, timestamp = reader.read_next()

        if topic != RGB_TOPIC:
            continue

        msg = deserialize_message(
            data,
            Image
        )

        image = ros_image_to_bgr(msg)

        # ----------------------------------------------------
        # Apply fixed homography
        # ----------------------------------------------------

        ipm = cv2.warpPerspective(
            image,
            homography,
            (
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT
            )
        )

        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(ipm)

        frame_count += 1

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        cv2.imshow(
            "RGB IPM Recording",
            ipm
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Stopped early by user.")
            break

        if frame_count % 100 == 0:

            print(
                f"Processed {frame_count} frames..."
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    writer.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("IPM recording complete.")
    print(f"Frames processed: {frame_count}")
    print()
    print("Output:")
    print(os.path.abspath(OUTPUT_VIDEO))
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()