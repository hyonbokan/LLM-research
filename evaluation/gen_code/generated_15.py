import pybgpstream
from collections import defaultdict

# Initialize BGP Stream
stream = pybgpstream.BGPStream(
    project="ris",
    collectors="rrc00",
    record_type="updates",
)

# Create a dictionary to store the customer cone for each ASN
customer_cone = defaultdict(set)

# Populate the customer cone dictionary
for rec in stream.records():
    for elem in rec:
        if 'as-path' in elem.fields and 'peer-asn' in elem.fields:
            as_path = elem.fields['as-path'].split()
            peer_asn = elem.fields['peer-asn']
            for asn in as_path:
                if asn!= peer_asn:
                    customer_cone[peer_asn].add(asn)

# Initialize a list to store the route leaks
route_leaks = []

# Iterate over the stream to detect route leaks
for rec in stream.records():
    for elem in rec:
        if 'as-path' in elem.fields and 'peer-asn' in elem.fields:
            as_path = elem.fields['as-path'].split()
            peer_asn = elem.fields['peer-asn']
            prefix = elem.fields['prefix']
            # Check if the peer ASN is not in its customer cone
            if peer_asn not in customer_cone or prefix not in customer_cone[peer_asn]:
                route_leaks.append((peer_asn, prefix, rec.time))

# Print the route leaks
for leak in route_leaks:
    print(f"Leaking ASN: {leak[0]}, Affected Prefix: {leak[1]}, Timestamp: {leak[2]}")