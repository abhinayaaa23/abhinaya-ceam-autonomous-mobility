import sqlite3

DB_PATH = r"C:\Users\abhin\Downloads\test\test4_20260902_134059\test4_20260902_134059_0.db3"

conn = sqlite3.connect(DB_PATH)

row = conn.execute(
    "SELECT id, name, type FROM topics WHERE name LIKE ?",
    ("%compressedDepth%",)
).fetchone()

print("Topic:", row)

sizes = conn.execute(
    "SELECT length(data) FROM messages WHERE topic_id=? LIMIT 5",
    (row[0],)
).fetchall()

print("First 5 message sizes:", sizes)

conn.close()