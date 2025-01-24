import pybgpstream

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    from_time="now-1m",
    until_time="now",
    collectors=["rrc00"],
    record_type="updates",
    filter="peer 15169 3356 7018"
)

# Process BGP records
for rec in stream.records():
    for elem in rec:
        if elem.type == "A":
            prefix = elem.fields['prefix']
            asn = elem.fields['as-path'].split()[-1]
            timestamp = elem.time
            print(f"Prefix: {prefix}, ASN: {asn}, Timestamp: {timestamp}")