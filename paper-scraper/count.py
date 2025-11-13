import ijson

filename = "polymer_metadata_cleaned.json"
count = 0

with open(filename, "r", encoding="utf-8") as f:
    for _ in ijson.items(f, "item"):
        count += 1

print(f"Total objects in file: {count}")
