from scapy.all import rdpcap
import sys
from src.PackerParser import NetworkTrafficTable, TCPTrafficTable
# 假设已经有了一个pcap文件
pcap_file = 'test/test.pcap'
packets = rdpcap(pcap_file)

# 实例化并使用NetworkTrafficTable
traffic_table = NetworkTrafficTable() 
tcp_table = TCPTrafficTable()
icmp_table = NetworkTrafficTable()
count = 0
for packet in packets:
    count += 1
    print('count:', count)  
    if 'ICMP' in packet or 'DNS' in packet or 'NTP' in packet:
        traffic_table.add_packet(packet)
        if 'ICMP' in packet:
            icmp_table.add_packet(packet)
    elif 'TCP' in packet:
        #tcp_table.add_packet(packet)
        pass
with open('test/t1.txt', 'w') as f:
    sys.stdout = f
    traffic_table.print_tables()
'''with open('test/icmp_dns_ntp_traffic_table.txt', 'w') as f:
    sys.stdout = f
    traffic_table.print_tables()
with open('test/tcp_traffic_table.txt', 'w') as f:
    sys.stdout = f
    tcp_table.print_tables()
# 生成RTT表
with open('test/icmp_dns_ntp_rtt_table.txt', 'w') as f:
    sys.stdout = f
    traffic_table.rtt_table.print_rtt()
with open('test/tcp_rtt_table.txt', 'w') as f:
    sys.stdout = f
    tcp_table.rtt_table.print_rtt()
with open('test/icmp_table.txt', 'w') as f:
    sys.stdout = f
    icmp_table.print_tables()
with open('test/icmp_rtt_table.txt', 'w') as f:
    sys.stdout = f
    icmp_table.rtt_table.print_rtt()
'''