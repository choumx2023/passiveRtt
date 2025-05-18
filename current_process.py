import argparse
import time
import os
import sys
import pickle
from src.PackerParser import NetworkTrafficTable, TCPTrafficTable
from scapy.layers.inet import TCP, IP, ICMP, UDP
from scapy.layers.dns import DNS
from scapy.main import load_contrib
load_contrib("ntp")
from scapy.contrib.ntp import NTP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
from src.Monitor import NetworkTrafficMonitor, CompressedIPNode, CompressedIPTrie
import logging
from typing import List
import contextlib
from scapy.all import rdpcap

def setup_logging(name: str = 'network_monitor') -> logging.Logger:
    '''
    params:
        name (str): The name of the logger.
    设置日志记录器。
    '''
    log_directory = "./logs"
    log_path = os.path.join(log_directory, f"{name}.log")
    print(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    return logger

def save_data_with_pickle(data: object, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data_with_pickle(filename: str) -> NetworkTrafficMonitor:
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def merge_monitors(file_list: List[str], output_dir: str) -> NetworkTrafficMonitor:
    if len(file_list) == 1:
        # 只有一个文件，直接返回这个监控器
        final_monitor = load_data_with_pickle(file_list[0])
        final_filename = os.path.join(output_dir, 'final_summary_monitor.pkl')
        save_data_with_pickle(final_monitor, final_filename)
        return final_monitor
    time1 = time.time()
    # 两两合并
    new_file_list = []
    for i in range(0, len(file_list), 2):
        if i + 1 < len(file_list):
            time1 = time.time()
            monitor1 = load_data_with_pickle(file_list[i])
            monitor2 = load_data_with_pickle(file_list[i + 1])
            monitor1.merge_monitor(monitor2)
            new_filename = os.path.join(output_dir, f'merged_monitor_{i//2}.pkl')
            save_data_with_pickle(monitor1, new_filename)
            new_file_list.append(new_filename)
            time2 = time.time()
            print(f'Merged {file_list[i]} and {file_list[i + 1]}, time: {time2 - time1}s')
        else:
            # 如果是奇数个，直接把最后一个加到新列表中
            new_file_list.append(file_list[i])

    # 递归继续合并
    return merge_monitors(new_file_list, output_dir)

def main(pcap_file: str, output_dir: str) -> None:
    print('Processing pcap file:', pcap_file)
    pcap_name = pcap_file.split('/')[-1].split('.')[0]
    output_dir = os.path.join(output_dir,  pcap_name)
    os.makedirs(output_dir, exist_ok=True)
    packets = rdpcap(pcap_file)
    logger: logging.Logger = setup_logging('current')
    monitor: NetworkTrafficMonitor = NetworkTrafficMonitor(name=pcap_name, check_anomalies=True, logger=logger)
    monitor.name = pcap_name
    traffic_table: NetworkTrafficTable = NetworkTrafficTable(monitor=monitor) 
    tcp_table: TCPTrafficTable = TCPTrafficTable(monitor=monitor)
    icmp_table: NetworkTrafficTable = NetworkTrafficTable(monitor=monitor)
    count = 0
    time1 = time.time()
    part_number = 0
    for packet in packets:
        count += 1
        print(count)

        # 确保数据包是IP数据包，并且访问协议字段
        if IP in packet:
            ip_layer = packet[IP]
            
            # ICMP协议编号为1
            if ip_layer.proto == 1:
                icmp_table.add_packet(packet[ICMP])

            # TCP协议编号为6
            elif ip_layer.proto == 6:
                tcp_table.add_packet(packet[TCP])

            # UDP协议编号为17
            elif ip_layer.proto == 17:
                # 进一步检查是否包含DNS或NTP
                if UDP in packet and (DNS in packet or NTP in packet):
                    from scapy.packet import Packet
                    assert isinstance(packet, Packet)
                    # 只将DNS或NTP层对象传递给add_packet
                    if DNS in packet:
                        traffic_table.add_packet(packet[DNS])
                    elif NTP in packet:
                        traffic_table.add_packet(packet[NTP])
            
            # 可以继续添加更多的协议编号检查
            # 例如ICMPv6的协议编号为58
            # ...

        # 其他特殊处理，例如IPv6特定的情况
        elif IPv6 in packet:
            if ICMPv6EchoRequest in packet:
                icmp_table.add_packet(packet[ICMPv6EchoRequest])
            elif ICMPv6EchoReply in packet:
                icmp_table.add_packet(packet[ICMPv6EchoReply])

        # every 10000 packets, print the number of packets processed and the time spent
        if count % 10000 == 0:
            
            time2 = time.time()
            print(f'Processed {count} packets, time: {time2 - time1}s')
            time1 = time2
        # every 2000000 packets, save the current monitor and create a new one
        if count % 2000000 == 0:
            part_number += 1
            # merge the current monitor into the summary monitor
            time3 = time.time()
            # create a new monitor for the next batch of packets
            traffic_table.clear_and_upload()
            tcp_table.clear_and_upload()
            icmp_table.clear_and_upload()
            save_data_with_pickle(monitor, os.path.join(output_dir, f'current_monitor_{part_number}.pkl'))
            monitor = NetworkTrafficMonitor(name='current', check_anomalies=True, logger=logger)
            monitor.name = 'current'
            traffic_table = NetworkTrafficTable(monitor=monitor)
            tcp_table = TCPTrafficTable(monitor=monitor)
            icmp_table = NetworkTrafficTable(monitor=monitor)
            time4 = time.time()
            time1 = time4
    # save the last part of the monitor
    if count % 2000000 != 0:
        part_number += 1
        traffic_table.flush_tables()
        icmp_table.flush_tables()
        tcp_table.flush_table()
        save_data_with_pickle(monitor, os.path.join(output_dir, f'current_monitor_{part_number}.pkl'))
    
    ## 读取各个part的monitor，合并成一个summary monitor
    summary_monitor = NetworkTrafficMonitor(name='summary', check_anomalies=True, logger=logger)
    summary_monitor.name = 'summary'
    # 自下而上合并
    file_list = [os.path.join(output_dir, f'current_monitor_{i}.pkl') for i in range(1, part_number + 1)]
    final_monitor: NetworkTrafficMonitor = merge_monitors(file_list, output_dir)
    final_monitor.name = 'final_summary'
    # 保存summary monitor
    save_data_with_pickle(final_monitor, os.path.join(output_dir, 'final_summary_monitor.pkl'))
    print('Summary monitor saved.')
    final_monitor.print_trees()
    final_monitor.analyze_traffic()

    with open(os.path.join(output_dir, 'icmp_dns_ntp_traffic_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            traffic_table.print_tables()
    with open(os.path.join(output_dir, 'tcp_traffic_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            tcp_table.print_tables()
    with open(os.path.join(output_dir, 'icmp_dns_ntp_rtt_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            traffic_table.rtt_table.print_rtt()
    with open(os.path.join(output_dir, 'tcp_rtt_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            tcp_table.rtt_table.print_rtt()
    with open(os.path.join(output_dir, 'icmp_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            icmp_table.print_tables()
    with open(os.path.join(output_dir, 'icmp_rtt_table.txt'), 'w') as f:
        with contextlib.redirect_stdout(f):
            icmp_table.rtt_table.print_rtt()

    # save_data_with_pickle(traffic_table, os.path.join(output_dir, 'icmp_dns_ntp_traffic.pkl'))
    # save_data_with_pickle(tcp_table, os.path.join(output_dir, 'tcp_traffic.pkl'))
    # save_data_with_pickle(icmp_table, os.path.join(output_dir, 'icmp_traffic.pkl'))
    # save_data_with_pickle(traffic_table.rtt_table, os.path.join(output_dir, 'icmp_dns_ntp_rtt.pkl'))
    # save_data_with_pickle(tcp_table.rtt_table, os.path.join(output_dir, 'tcp_rtt.pkl'))
    # save_data_with_pickle(icmp_table.rtt_table, os.path.join(output_dir, 'icmp_rtt.pkl'))
    save_data_with_pickle(final_monitor, os.path.join(output_dir, 'current_monitor.pkl'))
    
    print(f'All output files saved in directory: {output_dir}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a pcap file and generate traffic and RTT tables.')
    parser.add_argument('pcap_file', type=str, help='Path to the input pcap file')
    parser.add_argument('output_dir', type=str, help='Directory to save the output files')
    args = parser.parse_args()
    main(args.pcap_file, args.output_dir)
#python3 current_process.py ./test/test2.pcap ./data/rtt5
#python3 current_process.py ./test/test4.pcap ./data/rtt4
# python3 current_process.py ./test/t.pcap ./data/t > t.txt