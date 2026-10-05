import sqlite3

DB_PATH = r"C:\Users\abhin\Downloads\test\test4_20260902_134059\test4_20260902_134059_0.db3"

conn = sqlite3.connect(DB_PATH)

print("=" * 60)
print("SQLITE ROS 2 BAG INSPECTION")
print("=" * 60)

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print("\nTables:")
for table in tables:
    print(" -", table[0])

print("\nTopics:")
topics = conn.execute(
    "SELECT id, name, type FROM topics ORDER BY id"
).fetchall()

for topic_id, name, msg_type in topics:
    print(f"{topic_id:3} | {name} | {msg_type}")

conn.close()