import pybgpstream
from collections import defaultdict

# Initialize BGP Stream
stream = pybgpstream.BGPStream(
    project="ris",
    collectors="rrc00",
    from_time="now-2m",
    until_time="now",
    record_type="updates"
)

# Create a dictionary to store prefixes and their corresponding ASNs
prefix_asns = defaultdict(list)

# Iterate over the stream
for rec in stream.records():
    for elem in rec:
        if 'as-path' in elem.fields and 'prefix' in elem.fields:
            prefix = elem.fields['prefix']
            as_path = elem.fields['as-path'].split(' ')
            origin_as = as_path[-1]
            prefix_asns[prefix].append(origin_as)

# Find MOAS conflicts
moas_conflicts = {}
for prefix, asns in prefix_asns.items():
    if len(asns) > 1:
        moas_conflicts[prefix] = asns

# Print MOAS conflicts
for prefix, asns in moas_conflicts.items():
    print(f"Conflicting Prefix: {prefix}")
    for asn in asns:
        print(f"  ASN: {asn}")
        for rec in stream.records():
            for elem in rec:
                if 'as-path' in elem.fields and 'prefix' in elem.fields and elem.fields['prefix'] == prefix and elem.fields['as-path'].split(' ')[-1] == asn:
                    print(f"  Timestamp: {elem.time}")
    print()  # Empty line for better readability