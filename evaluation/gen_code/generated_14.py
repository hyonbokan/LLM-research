import pybgpstream
from collections import defaultdict

# Initialize BGP Stream
stream = pybgpstream.BGPStream(
    from_time="2022-01-01 00:00:00", until_time="2022-01-01 01:00:00",
    collectors=["rrc00"],
    record_type="updates"
)

# Dictionary to store AS paths and their lengths
as_paths = defaultdict(list)

# Loop through the stream records
for rec in stream.records():
    for elem in rec:
        if elem.type == "A":
            prefix = elem.fields.get('prefix', None)
            as_path = elem.fields.get('as-path', None)
            if prefix and as_path:
                as_path_length = len(as_path.split(" "))
                as_paths[prefix].append((as_path, as_path_length))

# Find the top 10 longest AS paths
longest_paths = sorted(as_paths.items(), key=lambda x: max([y[1] for y in x[1]]), reverse=True)[:10]

# Print the top 10 longest AS paths
for prefix, paths in longest_paths:
    max_length = max([y[1] for y in paths])
    longest_as_path = [y[0] for y in paths if y[1] == max_length][0]
    print(f"Prefix: {prefix}, Longest AS Path: {longest_as_path}, Length: {max_length}")