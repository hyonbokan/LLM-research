import pybgpstream

# Define the time range for data collection
start_time = "2023-03-01 00:00:00"
end_time = "2023-03-01 00:01:00"

# Initialize the BGPStream
stream = pybgpstream.BGPStream(
    from_time=start_time,
    until_time=end_time,
    collectors=["rrc00"],
    record_type="updates"
)

# Process BGP records
for rec in stream.records():
    for elem in rec:
        timestamp = elem.time
        collector = elem.collector
        raw_update = elem.fields
        print(f"Timestamp: {timestamp}, Collector: {collector}, Raw Update: {raw_update}")