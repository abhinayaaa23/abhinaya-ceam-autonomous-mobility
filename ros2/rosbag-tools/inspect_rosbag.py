from mcap.reader import make_reader

MCAP_PATH = r"C:\Users\abhin\Downloads\rosbag2_fresh3_0.mcap"

topics = {}
with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages():
        topics[channel.topic] = schema.name

print("Topics found:")
for topic, msg_type in topics.items():
    print(f"{topic} --> {msg_type}")
