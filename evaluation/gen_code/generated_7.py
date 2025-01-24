import pybgpstream
from collections import defaultdict
import time

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    from_time="now-2m",
    until_time="now",
    collectors=["rrc00"],
    record_type="updates"
)

# Dictionary to store the count of flaps for each prefix
prefix_flaps = defaultdict(int)

# Process BGP records
for rec in stream.records():
    for elem in rec:
        prefix = elem.fields['prefix']
        as_path = elem.fields['as-path'].split()
        asn = as_path[-1]

        # Check if the record is an announcement or withdrawal
        if elem.type == 'A' or elem.type == 'W':
            # Increment the flap count for the prefix
            prefix_flaps[prefix] += 1

            # Check if the prefix has flapped more than once
            if prefix_flaps[prefix] > 1:
                print(f"Route flapping detected for prefix {prefix} from ASN {asn}: {prefix_flaps[prefix]} flaps")

# Clean up
stream.disconnect()