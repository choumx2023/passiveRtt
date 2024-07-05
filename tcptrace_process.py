import re
import time
from datetime import datetime
import os
import sys
import pickle
import ipaddress
from src.RttTable import RTTTable

def convert_to_decimal(timestamp_str):
    # Use regex to clean up the timestamp string
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})", timestamp_str)
    if match:
        cleaned_timestamp_str = match.group(1)
    else:
        raise ValueError("Invalid timestamp format")

    # Convert string to datetime object
    dt = datetime.strptime(cleaned_timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

    # Convert datetime object to Unix timestamp
    unix_timestamp = time.mktime(dt.timetuple())

    # Convert Unix timestamp to decimal format, including microseconds
    decimal_timestamp = unix_timestamp + dt.microsecond / 1e6

    return decimal_timestamp

def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def extract_rtt(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    connections = []
    connection = {}
    for line in lines:
        if "host " in line:
            host_info = re.search(r'\s*host \w+: (.*)', line.strip()).group(1)
            ip_port = re.search(r'(.*):(\d+)', host_info).group(1, 2)
            if 'host_a' not in connection:
                connection['host_a'] = {'ip': ip_port[0].strip(), 'port': ip_port[1].strip()}
            else:
                connection['host_b'] = {'ip': ip_port[0].strip(), 'port': ip_port[1].strip()}
        elif 'packet' in line:
            time_match = re.search(r'(first packet|last packet):\s+(.*)', line.strip())
            if time_match:
                timestamp_str = time_match.group(2)
                if 'first_packet' not in connection:
                    connection['first_packet'] = convert_to_decimal(timestamp_str)
                else:
                    connection['last_packet'] = convert_to_decimal(timestamp_str)
        elif 'RTT samples' in line:
            rtt_samples = re.findall(r'(RTT samples):\s+(\d+)', line)
            for match in rtt_samples:
                key, value = match[0].lower().replace(' ', '_'), int(match[1])
                if key not in connection:
                    connection[key] = [value]
                else:
                    connection[key].append(value)
        else:
            matches = re.findall(r'(RTT samples|RTT avg|RTT min|RTT max):\s+(\d+\.\d+|\d+)', line)
            for match in matches:
                key, value = match[0].lower().replace(' ', '_'), float(match[1])
                if key not in connection:
                    connection[key] = [value]
                else:
                    connection[key].append(value)

        if 'host_b' in connection and 'rtt_avg' in connection:
            connections.append(connection)
            connection = {}

    return connections

def main(dir_path, output_dir):
    count = 0
    tcptrace_table = RTTTable()

    for filename in os.listdir(dir_path):
        min_time = -1
        max_time = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if filename.endswith(".txt"):
            count += 1
            connections = extract_rtt(f"{dir_path}/{filename}")
            for connection in connections:
                value = {'rtt_samples': connection['rtt_samples'], 'first_packet': connection['first_packet'], 'last_packet': connection['last_packet'], 'rtt_min': connection['rtt_min'], 'rtt_max': connection['rtt_max'], 'rtt_avg': connection['rtt_avg'], 'port_a' : connection['host_a']['port'], 'port_b' : connection['host_b']['port']}

                if connection['first_packet'] < min_time or min_time == -1:
                    min_time = connection['first_packet']
                if connection['last_packet'] > max_time:
                    max_time = connection['last_packet']
                src_ip = connection['host_a']['ip']
                dst_ip = connection['host_b']['ip']
                dir, reversed_dir = 'forward', 'backward'
                if ipaddress.ip_address(src_ip) > ipaddress.ip_address(dst_ip):
                    src_ip, dst_ip = dst_ip, src_ip
                    dir, reversed_dir = reversed_dir, dir
                if connection['rtt_samples'][0] != 0:
                    value1 = {'rtt': connection['rtt_min'][0], 'timestamp': connection['last_packet'], 'types': 'tcptrace'}
                    tcptrace_table.add_rtt_sample(src_ip=src_ip, dst_ip=dst_ip, rtt= value1['rtt'], timestamp=value1['timestamp'], types='tcptrace', direction=dir)
                if connection['rtt_samples'][1] != 0:
                    value2 = {'rtt': connection['rtt_min'][1], 'timestamp': connection['last_packet'], 'types': 'tcptrace'}
                    tcptrace_table.add_rtt_sample(src_ip=src_ip, dst_ip=dst_ip, rtt= value2['rtt'], timestamp=value2['timestamp'], types='tcptrace', direction=reversed_dir)
                tcptrace_table.max_time = max_time
                tcptrace_table.min_time = min_time
                # 比较这些连接在规定时间内，测量结果/baseline/tcptrace之间的差异
        else:
            continue
    save_data_with_pickle(tcptrace_table, os.path.join(output_dir, 'new_tcptrace.pkl'))
    with open(os.path.join(output_dir, 'new_tcptrace.txt'), 'w') as file:
        sys.stdout = file
        tcptrace_table.print_tcprtt()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    main(input_directory, output_directory)