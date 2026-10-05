import rosbag2_py

from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


BAG_PATH = (
    r"C:\Users\abhin\Downloads\test\test4_20260902_134059"
    r"\test4_20260902_134059_0.db3"
)


TF_TOPICS = {
    "/tf",
    "/tf_static",
}


def print_transform(transform, topic):

    t = transform.transform.translation
    q = transform.transform.rotation

    print()
    print("=" * 70)
    print(f"Topic: {topic}")
    print(f"Parent frame : {transform.header.frame_id}")
    print(f"Child frame  : {transform.child_frame_id}")

    print()
    print("Translation:")
    print(f"  x = {t.x:.6f}")
    print(f"  y = {t.y:.6f}")
    print(f"  z = {t.z:.6f}")

    print()
    print("Rotation quaternion:")
    print(f"  x = {q.x:.6f}")
    print(f"  y = {q.y:.6f}")
    print(f"  z = {q.z:.6f}")
    print(f"  w = {q.w:.6f}")


def main():

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
    print("Reading TF information...")

    seen = set()

    while reader.has_next():

        topic, data, timestamp = reader.read_next()

        if topic not in TF_TOPICS:
            continue

        msg = deserialize_message(
            data,
            TFMessage
        )

        for transform in msg.transforms:

            key = (
                topic,
                transform.header.frame_id,
                transform.child_frame_id
            )

            if key in seen:
                continue

            seen.add(key)

            print_transform(
                transform,
                topic
            )

    print()
    print("=" * 70)
    print(f"Unique transforms found: {len(seen)}")


if __name__ == "__main__":
    main()