import rosbag2_py
import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ============================================================
# Configuration
# ============================================================

BAG = r"C:\Users\abhin\Downloads\test\test4_20260902_134059"

DEPTH_TOPIC = "/zedx/zed_node/depth/depth_registered"


# ============================================================
# Open ROS 2 bag
# ============================================================

storage_options = rosbag2_py.StorageOptions(
    uri=BAG,
    storage_id="sqlite3"
)

converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = rosbag2_py.SequentialReader()

reader.open(
    storage_options,
    converter_options
)


# ============================================================
# Get topic information
# ============================================================

topics = reader.get_all_topics_and_types()

topic_types = {
    topic.name: topic.type
    for topic in topics
}

if DEPTH_TOPIC not in topic_types:
    print("ERROR: Depth topic not found!")
    print("\nAvailable topics:")

    for topic, msg_type in topic_types.items():
        print(f"  {topic} -> {msg_type}")

    raise SystemExit(1)


depth_msg_type_name = topic_types[DEPTH_TOPIC]

print("Depth topic:")
print(f"  {DEPTH_TOPIC}")

print("\nMessage type:")
print(f"  {depth_msg_type_name}")


# ============================================================
# Load ROS message type
# ============================================================

msg_type = get_message(depth_msg_type_name)


# ============================================================
# Search for first depth frame
# ============================================================

print("\nSearching for first depth frame...")

messages_checked = 0

while reader.has_next():

    topic, data, timestamp = reader.read_next()

    messages_checked += 1

    if topic != DEPTH_TOPIC:
        continue

    # Deserialize ROS Image message
    msg = deserialize_message(
        data,
        msg_type
    )

    print("\n========================================")
    print("FIRST DEPTH FRAME FOUND")
    print("========================================")

    print(f"Timestamp:      {timestamp}")
    print(f"Width:          {msg.width}")
    print(f"Height:         {msg.height}")
    print(f"Encoding:       {msg.encoding}")
    print(f"Step:           {msg.step}")
    print(f"Big endian:     {msg.is_bigendian}")
    print(f"Data bytes:     {len(msg.data)}")

    # ========================================================
    # Convert raw ROS image bytes → NumPy float32 array
    # ========================================================

    depth = np.frombuffer(
        msg.data,
        dtype=np.float32
    )

    depth = depth.reshape(
        (msg.height, msg.width)
    )

    print("\n========================================")
    print("DEPTH ARRAY")
    print("========================================")

    print(f"Shape:          {depth.shape}")
    print(f"Dtype:          {depth.dtype}")
    print(f"Min raw value:  {np.min(depth)}")
    print(f"Max raw value:  {np.max(depth)}")

    # ========================================================
    # Check valid / invalid values
    # ========================================================

    finite = np.isfinite(depth)

    nan_mask = np.isnan(depth)
    inf_mask = np.isinf(depth)

    valid = (
        finite &
        (depth > 0)
    )

    print("\n========================================")
    print("VALIDITY")
    print("========================================")

    print(f"Total pixels:           {depth.size}")
    print(f"Finite pixels:          {np.count_nonzero(finite)}")
    print(f"NaN pixels:             {np.count_nonzero(nan_mask)}")
    print(f"Inf pixels:             {np.count_nonzero(inf_mask)}")
    print(f"Valid positive pixels:  {np.count_nonzero(valid)}")

    # ========================================================
    # Statistics of valid depth values
    # ========================================================

    if np.any(valid):

        values = depth[valid]

        print("\n========================================")
        print("VALID DEPTH STATISTICS")
        print("========================================")

        print(f"Minimum:       {values.min()}")
        print(f"Maximum:       {values.max()}")
        print(f"Mean:          {values.mean()}")
        print(f"Median:        {np.median(values)}")

        print(
            f"1st percentile:  "
            f"{np.percentile(values, 1)}"
        )

        print(
            f"5th percentile:  "
            f"{np.percentile(values, 5)}"
        )

        print(
            f"95th percentile: "
            f"{np.percentile(values, 95)}"
        )

        print(
            f"99th percentile: "
            f"{np.percentile(values, 99)}"
        )

        # ====================================================
        # Sample values
        # ====================================================

        print("\n========================================")
        print("SAMPLE VALID DEPTH VALUES")
        print("========================================")

        print(values[:20])

    else:

        print(
            "\nWARNING: No positive finite depth "
            "values were found."
        )


    print("\n========================================")
    print(f"Messages checked: {messages_checked}")
    print("========================================")

    break


else:

    print("\nERROR: No depth frame was found in the bag.")


print("\nDone.")