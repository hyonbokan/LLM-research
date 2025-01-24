import pybgpstream

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    from_time="2023-03-01 00:00:00", until_time="2023-03-01 01:00:00",
    collectors=["rrc00"],
    record_type="updates"
)

# Dictionary to store withdrawn prefixes
withdrawn_prefixes = {}

# Process BGP records
for rec in stream.records():
    for elem in rec:
        if elem.type == "W":
            prefix = elem.fields.get('prefix', None)
            asn = elem.fields.get('as-path', None).split()[-1]
            withdrawn_prefixes[prefix] = asn

# Print withdrawn prefixes
print("Withdrawn Prefixes:")
for prefix, asn in withdrawn_prefixes.items():
    print(f"Timestamp: {rec.time}, Withdrawn Prefix: {prefix}, ASN: {asn}")