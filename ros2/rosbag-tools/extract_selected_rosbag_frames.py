from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import cv2
import os

MCAP_PATH = r"C:\Users\abhin\Downloads\rosbag2_fresh3_0.mcap"
CAMERA_TOPIC = "/camera/camera/color/image_raw"
OUTPUT_DIR = r"C:\Users\abhin\Downloads\extracted_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_count = 0
with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages():
        if channel.topic == CAMERA_TOPIC:
            frame_count += 1

print(f"Total camera frames: {frame_count}")

selected_indices = set(
    int(i * (frame_count - 1) / 9) for i in range(10)
)
print("Selected frames:", sorted(selected_indices))

decoder = DecoderFactory()

with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f)
    camera_index = 0

    for schema, channel, message in reader.iter_messages():
        if channel.topic != CAMERA_TOPIC:
            continue

        if camera_index in selected_indices:
            decode = decoder.decoder_for(
                channel.message_encoding, schema
            )
            decoded = decode(message.data)

            image = np.frombuffer(
                decoded.data, dtype=np.uint8
            )

            if decoded.encoding in ("rgb8", "bgr8"):
                image = image.reshape(
                    decoded.height, decoded.width, 3
                )
                if decoded.encoding == "rgb8":
                    image = cv2.cvtColor(
                        image, cv2.COLOR_RGB2BGR
                    )
            else:
                print("Unsupported encoding:", decoded.encoding)
                camera_index += 1
                continue

            filename = os.path.join(
                OUTPUT_DIR, f"frame_{camera_index:04d}.png"
            )
            cv2.imwrite(filename, image)
            print("Saved:", filename)

        camera_index += 1

print("Extraction complete!")
