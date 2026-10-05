import sqlite3
import struct
import numpy as np
import cv2
import os


DB_PATH = r"C:\Users\abhin\Downloads\test\test4_20260902_134059\test4_20260902_134059_0.db3"

DEPTH_TOPIC = "/zedx/zed_node/depth/depth_registered"

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)


def read_ros_string(data, offset):
    """
    Read a ROS 2 CDR string.

    Returns:
        string, new_offset
    """
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    raw = data[offset:offset + length]

    # ROS strings include a null terminator.
    value = raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    offset += length

    # CDR strings are padded to 4-byte alignment.
    offset = (offset + 3) & ~3

    return value, offset


def deserialize_image(data):
    """
    Deserialize the relevant fields from sensor_msgs/msg/Image.

    This parser is designed for the ROS 2 CDR representation
    used by the recorded bag.

    Returns:
        height, width, encoding, step, image_data
    """

    offset = 0

    # CDR encapsulation header
    offset += 4

    # std_msgs/Header
    # sec
    offset += 4

    # nanosec
    offset += 4

    # frame_id
    frame_id, offset = read_ros_string(data, offset)

    # Image height
    height = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # Image width
    width = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # Encoding
    encoding, offset = read_ros_string(data, offset)

    # is_bigendian
    is_bigendian = data[offset]
    offset += 1

    # Step
    step = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # Image data sequence length
    data_length = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    image_data = data[offset:offset + data_length]

    return (
        height,
        width,
        encoding,
        step,
        is_bigendian,
        frame_id,
        image_data,
    )


def main():

    print("=" * 60)
    print("TEST4 ZED DEPTH EXTRACTION")
    print("=" * 60)

    print(f"Bag: {DB_PATH}")
    print(f"Topic: {DEPTH_TOPIC}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    # Find topic ID
    result = cursor.execute(
        "SELECT id FROM topics WHERE name = ?",
        (DEPTH_TOPIC,)
    ).fetchone()

    if result is None:
        print("\nERROR: Depth topic not found.")
        conn.close()
        return

    topic_id = result[0]

    print(f"Topic ID: {topic_id}")

    # Get first depth message
    row = cursor.execute(
        """
        SELECT timestamp, data
        FROM messages
        WHERE topic_id = ?
        ORDER BY timestamp
        LIMIT 1
        """,
        (topic_id,)
    ).fetchone()

    if row is None:
        print("\nERROR: No messages found for depth topic.")
        conn.close()
        return

    timestamp, data = row

    print(f"Message timestamp: {timestamp}")
    print(f"Serialized message size: {len(data)} bytes")

    try:
        (
            height,
            width,
            encoding,
            step,
            is_bigendian,
            frame_id,
            image_data,
        ) = deserialize_image(data)

    except Exception as e:
        print("\nERROR while deserializing Image:")
        print(type(e).__name__, e)
        conn.close()
        return

    print("\nIMAGE INFORMATION")
    print("-" * 60)
    print(f"Resolution : {width} x {height}")
    print(f"Encoding   : {encoding}")
    print(f"Step       : {step} bytes")
    print(f"Frame ID   : {frame_id}")
    print(f"Data bytes : {len(image_data)}")

    expected_bytes = height * step

    print(f"Expected   : {expected_bytes} bytes")

    if len(image_data) != expected_bytes:
        print("\nWARNING: Image data size does not match height × step.")

    if encoding != "32FC1":
        print(f"\nWARNING: Expected 32FC1 but got {encoding}")

    # Convert raw bytes to float32 depth values.
    depth = np.frombuffer(
        image_data,
        dtype="<f4"
    ).reshape(height, width)

    # Valid depth = finite and positive.
    valid = np.isfinite(depth) & (depth > 0)

    total_pixels = depth.size
    valid_pixels = np.count_nonzero(valid)
    invalid_pixels = total_pixels - valid_pixels

    print("\nDEPTH STATISTICS")
    print("-" * 60)

    print(f"Total pixels   : {total_pixels:,}")
    print(f"Valid pixels   : {valid_pixels:,}")
    print(f"Invalid pixels : {invalid_pixels:,}")

    print(f"Valid %        : {100 * valid_pixels / total_pixels:.2f}%")
    print(f"Invalid %      : {100 * invalid_pixels / total_pixels:.2f}%")

    if valid_pixels > 0:

        valid_depth = depth[valid]

        print(f"\nMinimum depth  : {np.min(valid_depth):.3f}")
        print(f"Maximum depth  : {np.max(valid_depth):.3f}")
        print(f"Mean depth     : {np.mean(valid_depth):.3f}")
        print(f"Median depth   : {np.median(valid_depth):.3f}")

        print(f"\nPercentiles:")
        print(f"  5%  : {np.percentile(valid_depth, 5):.3f}")
        print(f"  25% : {np.percentile(valid_depth, 25):.3f}")
        print(f"  75% : {np.percentile(valid_depth, 75):.3f}")
        print(f"  95% : {np.percentile(valid_depth, 95):.3f}")

    # Save raw depth array
    npy_path = os.path.join(
        OUTPUT_DIR,
        "test4_depth_frame_000.npy"
    )

    np.save(npy_path, depth)

    print(f"\nSaved raw depth:")
    print(npy_path)

    # Create visualization
    visualization = np.zeros_like(depth, dtype=np.uint8)

    if valid_pixels > 0:

        valid_depth = depth[valid]

        # Robust visualization using 2nd–98th percentile.
        low = np.percentile(valid_depth, 2)
        high = np.percentile(valid_depth, 98)

        clipped = np.clip(depth, low, high)

        normalized = (
            (clipped - low) /
            (high - low + 1e-6) *
            255
        )

        visualization = normalized.astype(np.uint8)

        visualization[~valid] = 0

    colorized = cv2.applyColorMap(
        visualization,
        cv2.COLORMAP_TURBO
    )

    png_path = os.path.join(
        OUTPUT_DIR,
        "test4_depth_visualization_000.png"
    )

    cv2.imwrite(png_path, colorized)

    print("Saved visualization:")
    print(png_path)

    conn.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()