import sqlite3
import struct
import cv2
import numpy as np
import os


DB_PATH = r"C:\Users\abhin\Downloads\test\test4_20260902_134059\test4_20260902_134059_0.db3"

TOPIC_ID = 10

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)


def read_ros_string(data, offset):
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    value = data[offset:offset + length].rstrip(b"\x00").decode(
        "utf-8", errors="replace"
    )

    offset += length

    # CDR string alignment
    offset = (offset + 3) & ~3

    return value, offset


def deserialize_compressed_image(data):
    """
    Extract sensor_msgs/msg/CompressedImage:

    Header:
        sec
        nanosec
        frame_id

    format
    data[]
    """

    offset = 0

    # CDR encapsulation
    offset += 4

    # Header timestamp
    offset += 4       # sec
    offset += 4       # nanosec

    # Header frame_id
    frame_id, offset = read_ros_string(data, offset)

    # CompressedImage format
    format_name, offset = read_ros_string(data, offset)

    # data sequence length
    data_length = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    image_data = data[offset:offset + data_length]

    return frame_id, format_name, image_data


def main():

    print("=" * 60)
    print("TEST4 COMPRESSED DEPTH EXTRACTION")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """
        SELECT timestamp, data
        FROM messages
        WHERE topic_id = ?
        ORDER BY timestamp
        LIMIT 1
        """,
        (TOPIC_ID,)
    ).fetchone()

    if row is None:
        print("No compressed depth message found.")
        conn.close()
        return

    timestamp, serialized = row

    print(f"Timestamp: {timestamp}")
    print(f"Serialized size: {len(serialized):,} bytes")

    try:
        frame_id, format_name, compressed_data = (
            deserialize_compressed_image(serialized)
        )
    except Exception as e:
        print("\nERROR while parsing message:")
        print(type(e).__name__, e)
        conn.close()
        return

    print("\nMESSAGE")
    print("-" * 60)
    print(f"Frame ID : {frame_id}")
    print(f"Format   : {format_name}")
    print(f"Payload  : {len(compressed_data):,} bytes")

    # Decode compressed image using OpenCV.
    compressed_array = np.frombuffer(
        compressed_data,
        dtype=np.uint8
    )

    decoded = cv2.imdecode(
        compressed_array,
        cv2.IMREAD_UNCHANGED
    )

    if decoded is None:
        print("\nOpenCV could not decode the compressed image.")
        conn.close()
        return

    print("\nDECODED IMAGE")
    print("-" * 60)
    print(f"Shape : {decoded.shape}")
    print(f"Dtype : {decoded.dtype}")
    print(f"Min   : {np.min(decoded)}")
    print(f"Max   : {np.max(decoded)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_path = os.path.join(
        OUTPUT_DIR,
        "test4_compressed_depth_raw.png"
    )

    cv2.imwrite(raw_path, decoded)

    print(f"\nSaved decoded depth:")
    print(raw_path)

    # Visualization
    valid = np.isfinite(decoded) & (decoded > 0)

    if np.any(valid):

        values = decoded[valid].astype(np.float32)

        low = np.percentile(values, 2)
        high = np.percentile(values, 98)

        normalized = np.clip(
            (decoded.astype(np.float32) - low)
            / (high - low + 1e-6)
            * 255,
            0,
            255
        ).astype(np.uint8)

        normalized[~valid] = 0

        visualization = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_TURBO
        )

        vis_path = os.path.join(
            OUTPUT_DIR,
            "test4_compressed_depth_visualization.png"
        )

        cv2.imwrite(vis_path, visualization)

        print("Saved visualization:")
        print(vis_path)

    else:
        print("\nNo valid depth pixels detected.")

    conn.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()