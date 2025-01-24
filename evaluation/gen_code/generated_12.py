import pybgpstream

# Initialize BGP Stream
stream = pybgpstream.BGPStream(
    from_time="2023-02-15 00:00:00",
    until_time="2023-02-15 23:59:59",
    collectors=["rrc00"],
    record_type="updates"
)

# Dictionary to track prepending behavior
prepending_prefixes = {}

# Loop through the stream records
for rec in stream.records():
    for elem in rec:
        if elem.type == "A":
            prefix = elem.fields.get("prefix")
            as_path = elem.fields.get("as-path")
            as_path_list = as_path.split(" ")

            # Check for prepending
            prepended_count = 0
            for asn in as_path_list:
                if asn == str(elem.fields["as-path"].split(" ")[0]):
                    prepended_count += 1

            # Track prepending behavior
            if prefix not in prepending_prefixes:
                prepending_prefixes[prefix] = {"as_path": as_path, "prepended_count": prepended_count}
            else:
                prepending_prefixes[prefix]["prepended_count"] += prepended_count

# Output the results
for prefix, data in prepending_prefixes.items():
    print(f"Prefix: {prefix}")
    print(f"AS Path: {data['as_path']}")
    print(f"Prepended Entries: {data['prepended_count']}")
    print("\n")