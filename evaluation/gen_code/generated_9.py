import pybgpstream
from collections import defaultdict

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    from_time="2023-03-01 00:00:00",
    until_time="2023-03-01 00:01:00",
    collectors=["rrc00"],
    record_type="updates"
)

# Create a dictionary to store the frequency of each prefix
prefix_frequency = defaultdict(int)

# Iterate over the stream
for rec in stream.records():
    for elem in rec:
        if elem.type == "A":
            prefix = elem.fields.get('prefix', None)
            if prefix:
                prefix_frequency[prefix] += 1

# Get the top 10 most frequently announced prefixes
top_prefixes = sorted(prefix_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

# Print the top 10 most frequently announced prefixes
for prefix, frequency in top_prefixes:
    print(f"Prefix: {prefix}, Frequency: {frequency}")