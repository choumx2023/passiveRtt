import argparse
import os
import sys
import pickle
from scapy.all import rdpcap
from src.PackerParser import NetworkTrafficTable, TCPTrafficTable
from scapy.all import TCP, ICMP, DNS, NTP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply

def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def main(pcap_file, output_dir):
    packets = rdpcap(pcap_file)

    traffic_table = NetworkTrafficTable() 
    tcp_table = TCPTrafficTable()
    icmp_table = NetworkTrafficTable()
    count = 0

    for packet in packets:
        count += 1
        if ICMP in packet:
            icmp_table.add_packet(packet)
        elif IPv6 in packet and (ICMPv6EchoReply in packet or ICMPv6EchoRequest in packet):
            icmp_table.add_packet(packet)
        elif DNS in packet or NTP in packet:
            traffic_table.add_packet(packet)
        elif TCP in packet:
            tcp_table.add_packet(packet)

        if count % 10000 == 0:
            print(f'Processed {count} packets')

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'icmp_dns_ntp_traffic_table.txt'), 'w') as f:
        sys.stdout = f
        traffic_table.print_tables()
    with open(os.path.join(output_dir, 'tcp_traffic_table.txt'), 'w') as f:
        sys.stdout = f
        tcp_table.print_tables()
    with open(os.path.join(output_dir, 'icmp_dns_ntp_rtt_table.txt'), 'w') as f:
        sys.stdout = f
        traffic_table.rtt_table.print_rtt()
    with open(os.path.join(output_dir, 'tcp_rtt_table.txt'), 'w') as f:
        sys.stdout = f
        tcp_table.rtt_table.print_rtt()
    with open(os.path.join(output_dir, 'icmp_table.txt'), 'w') as f:
        sys.stdout = f
        icmp_table.print_tables()
    with open(os.path.join(output_dir, 'icmp_rtt_table.txt'), 'w') as f:
        sys.stdout = f
        icmp_table.rtt_table.print_rtt()

    save_data_with_pickle(traffic_table, os.path.join(output_dir, 'icmp_dns_ntp_traffic.pkl'))
    save_data_with_pickle(tcp_table, os.path.join(output_dir, 'tcp_traffic.pkl'))
    save_data_with_pickle(icmp_table, os.path.join(output_dir, 'icmp_traffic.pkl'))
    save_data_with_pickle(traffic_table.rtt_table, os.path.join(output_dir, 'icmp_dns_ntp_rtt.pkl'))
    save_data_with_pickle(tcp_table.rtt_table, os.path.join(output_dir, 'tcp_rtt.pkl'))
    save_data_with_pickle(icmp_table.rtt_table, os.path.join(output_dir, 'icmp_rtt.pkl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a pcap file and generate traffic and RTT tables.')
    parser.add_argument('pcap_file', type=str, help='Path to the input pcap file')
    parser.add_argument('output_dir', type=str, help='Directory to save the output files')
    args = parser.parse_args()

    main(args.pcap_file, args.output_dir)
#python3 current_process.py ./test/test2.pcap ./data/rtt5