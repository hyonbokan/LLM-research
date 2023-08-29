import json

jsonfile = "/home/hb/fine-tuning-alpaca/regen.json"

# Load JSON data from file
with open(jsonfile, "r") as f:
    json_data = json.load(f)

# Calculate the number of elements in the JSON object
num_elements = len(json_data)

print("Number of elements in the JSON object:", num_elements)