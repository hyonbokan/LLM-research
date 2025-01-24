import pybgpstream
import time

# Define the prefix and the collection period
prefix = "1.2.3.4/24"
collection_period = 60  # 1 minute

# Initialize the stream
stream = pybgpstream.BGPStream(
    project="ris",
    collectors=["rrc00"],
    record_type="updates",
    filter="prefix %s" % prefix
)

# Initialize the previous AS path
previous_as_path = None

# Start the stream
start_time = time.time()
for rec in stream.records():
    if time.time() - start_time > collection_period:
        break
    for elem in rec:
        if elem.type == "A":
            as_path = elem.fields.get('as-path', '')
            if previous_as_path and as_path!= previous_as_path:
                print(f"Prefix: {prefix}")
                print(f"Previous AS Path: {previous_as_path}")
                print(f"New AS Path: {as_path}")
                print("------------------------")
            previous_as_path = as_path

# Close the stream
stream.disconnect()