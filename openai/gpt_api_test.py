import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt4_output(prompt):
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "write a simple python code to solve two sum"}
            ],
            max_tokens=2000,
            temperature=0.7
        )

    # Extract the assistant's reply
    assistant_reply_content = response.choices[0].message.content
    return assistant_reply_content

def extract_code_from_reply(assistant_reply_content):
    code_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_pattern, assistant_reply_content, re.DOTALL)
    if match:
        code = match.group(1)
        return code
    else:
        print("No code block found in the assistant's reply.")
        return None
    
if __name__ == "__main__":
    # Your prompt
    SYSTEM_PROMPT = '''
    You are tasked with generating Python scripts that perform various BGP analysis tasks using the pybgpstream library. Please adhere to the following guidelines when writing the code:

    - Main Loop Processing:
    Do not use any filter attributes like stream.add_filter() or set filter parameters when initializing BGPStream.
    All filtering and processing should occur within the main loop where you iterate over records and elements.

    - Script Structure:
    Start by importing necessary libraries, including pybgpstream and any others required for the task (e.g., datetime, collections).
    Define a main function or functions that encapsulate the core logic.
    Include a __main__ block or a usage example to demonstrate how to run the script.

    - Key Processing Guidelines:
    * Time Format: Define the time range as strings in the following format: from_time = "YYYY-MM-DD HH:MM:SS"
    until_time = "YYYY-MM-DD HH:MM:SS"

    * Stream Initialization: Use these time parameters during BGPStream initialization:
    stream = pybgpstream.BGPStream(
        from_time=from_time,
        until_time=until_time,
        record_type="updates",
        collectors=collectors
    )

    * Iterating Over Records and Elements:
    for rec in stream.records(): for elem in rec: # Processing logic goes here

    * Accessing Element Attributes:
    Timestamp: elem_time = datetime.utcfromtimestamp(elem.time)

    Element Type (Announcement or Withdrawal): elem_type = elem.type # 'A' for announcements, 'W' for withdrawals

    Fields Dictionary: fields = elem.fields

    Prefix: prefix = fields.get("prefix") if prefix is None: continue # Skip if prefix is not present

    AS Path: as_path_str = fields.get('as-path', "") as_path = as_path_str.split()

    Peer ASN and Collector: peer_asn = elem.peer_asn collector = rec.collector

    Communities: communities = fields.get('communities', [])

    * Filtering Logic Within the Loop:
    Filtering for a Specific ASN in AS Path: target_asn = '64500' if target_asn not in as_path: continue # Skip if target ASN is not in the AS path

    Filtering for Specific Prefixes: target_prefixes = ['192.0.2.0/24', '198.51.100.0/24'] if prefix not in target_prefixes: continue # Skip if prefix is not in target prefixes

    * Processing Key Values and Attributes:
    Counting Announcements and Withdrawals: if elem_type == 'A': announcements[prefix] += 1 elif elem_type == 'W': withdrawals[prefix] += 1

    Detecting AS Path Changes: if prefix in prefix_as_paths: if as_path != prefix_as_paths[prefix]: # AS path has changed prefix_as_paths[prefix] = as_path else: prefix_as_paths[prefix] = as_path

    Analyzing Community Attributes: for community in communities: community_str = f"{community[0]}:{community[1]}" community_counts[community_str] += 1

    Calculating Statistics (e.g., Average MED): med = fields.get('med') if med is not None: try: med_values.append(int(med)) except ValueError: pass # Ignore invalid MED values

    * Detecting Hijacks: Compare the observed origin AS with the expected origin AS for target prefixes:
    expected_origins = {'192.0.2.0/24': '64500', '198.51.100.0/24': '64501'}
    if prefix in expected_origins:
        observed_origin = as_path[-1] if as_path else None
        expected_origin = expected_origins[prefix]
        if observed_origin != expected_origin:
            # Potential hijack detected
            print(f"Possible hijack detected for {prefix}: expected {expected_origin}, observed {observed_origin}")

    * Detecting Outages:
    * Monitor for withdrawals of prefixes without re-announcements:
    Keep track of withdrawn prefixes and their timestamps
    if elem_type == 'W':
        withdrawals[prefix] = elem_time
    elif elem_type == 'A':
        # Remove from withdrawals if re-announced
        if prefix in withdrawals:
            del withdrawals[prefix]
    Check if prefix remains withdrawn for a certain period (e.g., 30 minutes)
    for prefix, withdrawal_time in list(withdrawals.items()):
        if elem_time - withdrawal_time > timedelta(minutes=30):
            # Outage detected for prefix
            print(f"Outage detected for {prefix} starting at {withdrawal_time}")
            del withdrawals[prefix]

    * Detecting MOAS (Multiple Origin AS) Conflicts: Monitor prefixes announced by multiple origin ASNs
    origin_asn = as_path[-1] if as_path else None
    if origin_asn:
        if prefix not in prefix_origins:
            prefix_origins[prefix] = set()
        prefix_origins[prefix].add(origin_asn)
        if len(prefix_origins[prefix]) > 1:
            # MOAS conflict detected
            origins = ', '.join(prefix_origins[prefix])
            print(f"MOAS conflict for {prefix}: announced by ASNs {origins}")

    * Analyzing AS Path Prepending: Detect AS path prepending by identifying consecutive repeated ASNs in the AS path:
    last_asn = None
    consecutive_count = 1
    for asn in as_path:
        if asn == last_asn:
            consecutive_count += 1
        else:
            if consecutive_count > 1:
                prepending_counts[last_asn] += consecutive_count - 1
            consecutive_count = 1
        last_asn = asn
    Check for prepending at the end of the path
    if consecutive_count > 1 and last_asn:
        prepending_counts[last_asn] += consecutive_count - 1
        
    * Handling IP Addresses and Prefixes:
    Validating and Parsing IP Prefixes: import ipaddress try: network = ipaddress.ip_network(prefix) except ValueError: continue # Skip invalid prefixes

    Here is your tasks:
    '''
    user_query = "list bgp messages for as3356 from rrc00 between oct 28 13pm and 13:01pm 2024"
    prompt = SYSTEM_PROMPT + user_query
    output = get_gpt4_output(prompt)
    print(output)
    if output:
        code = extract_code_from_reply(output)
        if code:
            # Save the code to a file
            with open('bgp_script.py', 'w') as f:
                f.write(code)
            print("Code saved to bgp_script.py")
        else:
            print("Failed to extract code from the assistant's reply.")