import pybgpstream

stream = pybgpstream.BGPStream(
    from_time="2022-01-01 13:00:00",
    until_time="2022-01-01 13:30:00",
    collectors=["route-views.eqix"],
    record_type="updates",
    filter="ipversion 4"
)

total_updates = 0

for elem in stream:
    total_updates += 1

print(total_updates)