from scapy.all import IP, ICMP, UDP, DNS, NTP, TCP
from scapy.layers.inet6 import ICMPv6EchoRequest, ICMPv6EchoReply, IPv6
from typing import Union, List
from .CuckooHashTable import CuckooHashTable, ListBuffer, TcpState
from .RttTable import RTTTable
from utils.utils import extract_ip
import time
from .Monitor import NetworkTrafficMonitor
import ipaddress
import json
# CuckooHashTable key要求是字典
def ip_compare(src_ip, dst_ip):
    ip1 = ipaddress.ip_address(src_ip)
    ip2 = ipaddress.ip_address(dst_ip)
    
    if ip1 > ip2:
        return (src_ip, dst_ip)
    else:
        return (dst_ip, src_ip)
class NetworkTrafficTable(CuckooHashTable):
    '''
    A table to store network traffic data, including ICMP, DNS, and NTP packets.
    def __init__(self, initial_size = 10013, buffersize = 1000) -> None:
        self.size = initial_size
        self.buffersize = buffersize
        self.tables = [[False, 'no_stage'] * self.size for _ in range(3)]
        self.values = [[ListBuffer(buffersize) for _ in range(self.size)] for _ in range(3)]
        self.num_items = 0
        self.rehash_threshold = 0.6
        self.max_rehash_attempts = 5
    '''
    def __init__(self, monitor : NetworkTrafficMonitor = NetworkTrafficMonitor(), initial_size=10013):
        '''
        params:
            initial_size (int): The initial size of the hash table.
            monitor (NetworkTrafficMonitor): The monitor instance to record network activities.
            initial_size (int): 哈希表的初始大小。
        '''
        super().__init__(initial_size, buffersize = 50, type = 'TCP') 
        self.rtt_table = RTTTable()
        self.size : int
        self.buffersize : int
        self.tables : list[list[bool, str]]
        self.values : list[list[ListBuffer]]
        self.tcp_state : list[TcpState]
        self.net_monitor = monitor

    def replace_monitor(self, monitor : NetworkTrafficMonitor):
        '''
        To decrease the complexity of the append function, we update the summary monitor and replace the monitor.
        params:
            monitor (NetworkTrafficMonitor): The new monitor instance to replace the old one.
        '''
        self.net_monitor = monitor
        
    def add_packet(self, packet : Union[ICMP, ICMPv6EchoRequest, ICMPv6EchoReply, DNS, NTP, TCP, UDP]):
        if ICMP in packet:
            self.process_icmp_packet(packet)
        elif DNS in packet and UDP in packet:
            self.process_dns_packet(packet)
        elif NTP in packet:
            self.process_ntp_packet(packet)
        else:
            pass
    def process_icmp_packet(self, packet):
        '''
        This function processes ICMP packets, tries to match requests and responses, and then updates the RTT table.
        params:
            packet (Union[ICMP, ICMPv6EchoRequest, ICMPv6EchoReply]): The ICMP packet to process.
        function:
            for RttTable:
                add_rtt_sample
            for NetworkTrafficMonitor:
                add_ip_and_record_activity
        '''
        src_ip, dst_ip = extract_ip(packet)
        timestamp = packet.time
        # 确定是 ICMP 还是 ICMPv6，并提取相应的字段
        if ICMP in packet:
            icmp_id = packet[ICMP].id
            icmp_seq = packet[ICMP].seq
            icmp_type = packet[ICMP].type  # 8为请求，0为响应
            protocol = 'ICMP'
            self.net_monitor.add_ip_and_record_activity(src_ip, protocol, 'Request' if icmp_type == 8 else 'Response', 1, float(timestamp))
            self.net_monitor.add_ip_and_record_activity(dst_ip, protocol, 'Request' if icmp_type == 8 else 'Response', 1, float(timestamp))
        elif IPv6 in packet and (ICMPv6EchoRequest in packet or ICMPv6EchoReply in packet):
            protocol = 'ICMPv6'
            icmp_id = None
            icmp_seq = None
            icmp_type = None
            if ICMPv6EchoRequest in packet:
                icmp_id = packet[ICMPv6EchoRequest].id
                icmp_seq = packet[ICMPv6EchoRequest].seq
                icmp_type = 128  # Echo Request (ICMPv6 type 128)
            elif ICMPv6EchoReply in packet:
                icmp_id = packet[ICMPv6EchoReply].id
                icmp_seq = packet[ICMPv6EchoReply].seq
                icmp_type = 129  # Echo Reply (ICMPv6 type 129)
            if icmp_type is not None:
                self.net_monitor.add_ip_and_record_activity(src_ip, protocol, 'Request' if icmp_type == 128 else 'Response', 1, float(timestamp))
                self.net_monitor.add_ip_and_record_activity(dst_ip, protocol, 'Request' if icmp_type == 128 else 'Response', 1, float(timestamp))
            print("find ICMPv6!")
        else:
            print("Not ICMP or ICMPv6", value)
            return  # 不是我们感兴趣的包
        if float(timestamp) > 1714302731.6 and float(timestamp) < 17143025731.9:
            print("find ICMP! and pass return ,between 1714302731.6 and 17143025731.9",float(timestamp))
        key = {'ip': (src_ip, dst_ip), 'protocol': protocol}
        value = {'timestamp': timestamp, 'type': icmp_type, 'seq': icmp_seq, 'id': icmp_id}

        if icmp_type not in [0, 8, 128, 129]:
            print("icmp_type not in [0, 8, 128, 129]", icmp_type)
            print(value)
            return  # 只处理请求和响应报文
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': protocol}
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values = None
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            if icmp_type in [8, 128]:  # ICMP或ICMPv6请求
                # 直接插入，等待响应
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element=value, condition1=condition1, condition2=condition2, is_add=True)
            elif icmp_type in [0, 129]:  # ICMP或ICMPv6响应
                # 尝试找到匹配的请求
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: x['type'] in [8, 128] and x['seq'] == y['seq'] and x['id'] == y['id']
                condition2 = lambda x, y: False
                print(self.values[table_num][index].__state__())
                prior_value = target_values.process_element(new_element=value, condition1=condition1, condition2=condition2, is_add=False)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    self.net_monitor.add_or_update_ip_with_rtt(src_ip, protocol, 'Response', float(rtt*1000), float(timestamp))
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, direction=value['direction'], types = protocol)
    
    '''
    def process_icmp_packet(self, packet: Union[ICMP, ICMPv6EchoRequest, ICMPv6EchoReply]):
        """处理ICMP数据包，尝试匹配请求和响应，然后更新RTT表"""
        src_ip, dst_ip = extract_ip(packet)
        icmp_id = packet[ICMP].id
        icmp_seq = packet[ICMP].seq
        icmp_type = packet[ICMP].type  # 8为请求，0为响应
        timestamp = packet.time
        key = {'ip' : (src_ip, dst_ip), 'protocol': 'ICMP'}
        value = {'timestamp': timestamp, 'type': icmp_type, 'seq': icmp_seq}

        if icmp_type != 0 and icmp_type != 8:
            return
        table_num, index = self.lookup(key)
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            if icmp_type == 8:  # ICMP请求
                # 直接插入，等待响应
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element( new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif icmp_type == 0:  # ICMP响应
                # 尝试找到匹配的请求
                #print(key)
                #print(value)
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: x['type'] == 8 and x['seq'] == y['seq']
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    #print(f"prior_value: {prior_value}")
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'ICMP')
                        #print(f"RTT: {rtt}")
                # 可选择删除请求记录或保留以支持多次测量
                # self.delete(key)
'''
    def process_dns_packet(self, packet):
        '''
        This function processes DNS packets, tries to match requests and responses, and then updates the RTT table.
        params:
            packet (DNS): The DNS packet to process.
        '''
        src_ip, dst_ip = extract_ip(packet)
        dst_port = packet[UDP].dport
        src_port = packet[UDP].sport
        dns_id = packet[DNS].id
        qr = packet[DNS].qr

        timestamp = packet.time
        key = {'ip':(src_ip, dst_ip),'port':(src_port, dst_port), 'protocol': 'DNS'}
        value = {'timestamp': timestamp, 'dns_id': dns_id, 'is_qr': qr, 'dns_details':[packet[DNS].qdcount, packet[DNS].ancount, packet[DNS].nscount, packet[DNS].arcount]}
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'DNS', 'port': (src_port, dst_port)}
        self.net_monitor.add_ip_and_record_activity(src_ip, 'DNS', 'Request' if qr == 0 else 'Response', 1, float(timestamp))
        self.net_monitor.add_ip_and_record_activity(dst_ip, 'DNS', 'Request' if qr == 0 else 'Response', 1, float(timestamp))
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            if qr == 0:  # Query
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif qr == 1:  # Response
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['dns_id'] == new_item['dns_id'] and existing_item['is_qr'] == 0 and new_item['is_qr'] == 1
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = False)
                if prior_value is not None:
                    # find the matching request and calculate RTT
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp          
                    self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'DNS', 'Response', float(rtt*1000), float(timestamp))
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'DNS', direction=value['direction'], extra_data=[value['dns_details'], prior_value['dns_details']])
                        #print(f"RTT: {rtt}")
                    #self.delete(request_key)                    

    def process_ntp_packet(self, packet: NTP):
        src_ip, dst_ip = extract_ip(packet)
        dst_port = packet[UDP].dport
        src_port = packet[UDP].sport
        ntp_mode = packet[NTP].mode
        if ntp_mode not in [3, 4]:
            return
        timestamp = packet.time
        orig_timestamp = packet[NTP].orig
        recv_timestamp = packet[NTP].recv
        sent_timestamp = packet[NTP].sent
        ref_timestamp = packet[NTP].ref
        self.net_monitor.add_ip_and_record_activity(src_ip, 'NTP', 'Request' if ntp_mode == 3 else 'Response', 1, float(timestamp))
        self.net_monitor.add_ip_and_record_activity(dst_ip, 'NTP', 'Request' if ntp_mode == 3 else 'Response', 1, float(timestamp))
        key = {'ip':(src_ip, dst_ip),'port':(src_port, dst_port), 'protocol': 'NTP'}
        value = {'timestamp': timestamp, 'mode': ntp_mode, 'orig': orig_timestamp, 'ref': ref_timestamp, 'recv': recv_timestamp, 'sent': sent_timestamp}
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'NTP', 'port': (src_port, dst_port)}
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            if ntp_mode == 3:
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif ntp_mode == 4:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['mode'] == 3 and new_item['mode'] == 4 and new_item['ref'] == existing_item['orig']
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = False)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    self.net_monitor.add_ip_and_record_activity(src_ip, 'NTP', 'Response', 1, float(timestamp))
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'NTP', direction=value['direction'])
                        #print(f"RTT: {rtt}")
                #self.delete(request_key)

    def process_udp_packet(self, packet):
        src_ip, dst_ip = extract_ip(packet)
        port = packet[UDP].sport
        protocol = 'UDP'
        timestamp = packet.time
        key = (src_ip, dst_ip, port, protocol)
        self.insert(key, {'timestamp': timestamp})

    def calculate_rtt(self, packet, prior_value):
        rtt = packet.time - prior_value['timestamp']
        return rtt
    

class TCPTrafficTable(CuckooHashTable):
    '''
    This class is used to store TCP traffic data, including SYN, SYN-ACK, and PSH packets.
    '''
    def __init__(self, monitor : NetworkTrafficMonitor, initial_size=1019, rtt_table=None):
        '''
        This function initializes the TCPTrafficTable, add the monitor instance, and the RTT table instance.
        '''
        super().__init__(type = 'TCP')
        self.rtt_table = rtt_table or RTTTable()  # RTT表实例，用于存储RTT计算结果
        self.tables : list[list[bool, str]]
        self.values : list[list[ListBuffer]]
        self.tcp_state : list[list[TcpState]]
        self.net_monitor = monitor
    def delete_entry(self, table_num, index):
        '''
        This function deletes the element from the table.
        and update the monitor record.
        params:
            table_num (int): The table number.
            index (int): The index of the element to delete.
        '''
        record = self.tcp_state[table_num][index].get_flow_record()
        ip_pair = self.tables[table_num][index][0]['ip']
        if record['max_length'][0] != -1 and record['max_length'][1] != -1:
            self.net_monitor.add_flow_record(ip_pair, record)
        self.values[table_num][index] = ListBuffer(self.buffersize)
        self.tables[table_num][index] = None
        self.tcp_state : list[ list [TcpState]  ]
        if self.type == 'TCP':
            self.tcp_state[table_num][index].clear()
        # 此处应该更新monitor的记录
    def add_packet(self, packet : Union[TCP]):
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            syn_flag = (tcp_flags & 0x2) >> 1
            psh_flag = (tcp_flags & 0x8) >> 3
            ts_val, _ = self.extract_tcp_options(packet)
            if syn_flag:
                self.process_syn_packet(packet)
            elif ts_val is not None:
                self.process_timestamp_packet(packet)
            elif psh_flag:
                self.process_normal_packet(packet, "PSH")
            else:
                self.process_normal_packet(packet)
            # 尝试找到匹配的数据包并计算RTT
            
    def extract_tcp_options(self, packet : TCP):
        '''
        This function extracts the TCP options from the packet. If the packet contains the Timestamp option, it returns the timestamp value and echo reply.
        params:
            packet (TCP): The TCP packet to extract the options from.
        returns:
            ts_val (int): The timestamp value.
            ts_ecr (int): The timestamp echo reply.
        '''
        ts_val, ts_ecr = None, None
        if TCP in packet:
            for option in packet[TCP].options:
                if option[0] == 'Timestamp':
                    ts_val, ts_ecr = option[1]
                    break
        return ts_val, ts_ecr
    # 处理SYN数据包
    def process_syn_packet(self, packet : TCP):
        '''
        This function processes SYN packets, tries to match requests and responses, and then updates the RTT table.
        params:
            packet (TCP): The TCP packet to process.
        returns:
            None
        function:
            for RttTable:
                add_rtt_sample
            for NetworkTrafficMonitor:
                add_ip_and_record_activity
        '''
        self.tcp_state : list[list[TcpState]]
        src_ip, dst_ip = extract_ip(packet)     
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) != 0
        syn_flag = (tcp_flags & 0x02) != 0
        fin_flag = (tcp_flags & 0x01) != 0

        # 获取IP头部长度和TCP头部长度
        if ":" in src_ip:
            ip_header_len = 40
        else:
            ip_header_len = packet[IP].ihl * 4
        tcp_header_len = packet[TCP].dataofs * 4
        if ":" in src_ip:
            ip_total_length = packet[IPv6].plen
        else:
            ip_total_length = packet[IP].len  # IP总长度包括IP头和TCP段
        tcp_payload_len = ip_total_length - ip_header_len - tcp_header_len  # TCP负载长度
        # 序列号计算
        flags_count = int(syn_flag) + int(fin_flag)
        next_seq = packet[TCP].seq + tcp_payload_len + flags_count
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        timestamp = packet.time
        value = {'timestamp': packet.time, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack, 'SYN':1, 'ACK': ack_flag, 'length': tcp_payload_len, 'next_seq': next_seq}
        
        
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        table_num, index = self.lookup(key_temp)
        if syn_flag and not ack_flag and table_num is not None: # 和之前的记录冲突了
            self.delete_entry(table_num, index)
            table_num = None
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            # ACK & SYN
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            
            is_valid, packet_type = self.tcp_state[table_num][index].update_state(value)
            if ack_flag:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['SYN'] == 1 and existing_item['ACK'] == 0 and existing_item['seq'] == new_item['ack'] - 1 and existing_item['direction'] != new_item['direction']
                condition2 = lambda existing_item, new_item: False
                prior_value = target_values.process_tcp_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'SYN-ACK', float(rtt*1000), float(timestamp))
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'SYN', direction=value['direction'])
            # SYN
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_tcp_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
        self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'SYN-ACK' if ack_flag else 'SYN', 1, float(timestamp))
        self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'SYN-ACK' if ack_flag else 'SYN', 1, float(timestamp))
        
    # 处理Timestamp数据包
    def process_timestamp_packet(self, packet : TCP):
        '''
        This function processes Timestamp packets, tries to match requests and responses, and then updates the RTT table.
        params:
            packet (TCP): The TCP packet to process.
        returns:
            None
        function:
            for RttTable:
                add_rtt_sample
            for NetworkTrafficMonitor:
                add_ip_and_record_activity
        '''
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) != 0
        syn_flag = (tcp_flags & 0x02) != 0
        fin_flag = (tcp_flags & 0x01) != 0
        if ":" in src_ip:
            ip_header_len = 40
        else:
            ip_header_len = packet[IP].ihl * 4
        tcp_header_len = packet[TCP].dataofs * 4
        if ":" in src_ip:
            ip_total_length = packet[IPv6].plen
        else:
            ip_total_length = packet[IP].len  # IP总长度包括IP头和TCP段
        tcp_payload_len = ip_total_length - ip_header_len - tcp_header_len  # TCP负载长度

        # 序列号计算，包括SYN和FIN标志的影响
        flags_count = int(syn_flag) + int(fin_flag)
        next_seq = packet[TCP].seq + tcp_payload_len + flags_count
        # 序列号计算
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        ts_val, ts_ecr = self.extract_tcp_options(packet)
        timestamp = packet.time
        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': ack_flag , 'ts_val': ts_val, 'ts_ecr': ts_ecr, 'length': tcp_payload_len, 'ACK': int(ack_flag), 'SYN': int(syn_flag), 'next_seq': next_seq, 'FIN': int(fin_flag)}
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'Timestamp', 1, float(timestamp))
        self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'Timestamp', 1, float(timestamp))
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            

            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            self.tcp_state[table_num][index].update_state(value)
            if fin_flag:
                self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'FIN', 1, float(timestamp))
                ip_pair = self.tables[table_num][index][0]['ip']
                record =  self.tcp_state[table_num][index].get_flow_record()
                if record['fin_sign'] == 3 and record['max_length'][0] != -1 and record['max_length'][1] != -1:
                    self.net_monitor.add_flow_record(ip_pairs=ip_pair, flow_record= record)
            if ack_flag:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item.get('ts_val') is not None and existing_item['ts_val'] == new_item['ts_ecr'] and existing_item['direction'] != new_item['direction']
                condition2 = lambda x, y: False
                prior_value = target_values.process_tcp_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'Timestamp', float(rtt*1000), float(timestamp))
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'Timestamp', direction=value['direction'])
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_tcp_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
    def process_reset_packet(self, packet : TCP, additional_info :str = None):
        '''
        This function is used to process the reset packet.
        params:
            packet (scapy.Packet): The packet to be processed.
            additional_info (str): The additional information to be processed.
        return:
            None
        function:
            for NetworkTrafficMonitor:
                add_ip_and_record_activity
        '''
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) != 0
        syn_flag = (tcp_flags & 0x02) != 0
        fin_flag = (tcp_flags & 0x01) != 0

        # 获取IP头部长度和TCP头部长度
        if ":" in src_ip:
            ip_header_len = 40
        else:
            ip_header_len = packet[IP].ihl * 4
        tcp_header_len = packet[TCP].dataofs * 4
        if ":" in src_ip:
            ip_total_length = packet[IPv6].plen
        else:
            ip_total_length = packet[IP].len  # IP总长度包括IP头和TCP段
        tcp_payload_len = ip_total_length - ip_header_len - tcp_header_len  # TCP负载长度

        # 序列号计算，包括SYN和FIN标志的影响
        flags_count = int(syn_flag) + int(fin_flag)
        next_seq = packet[TCP].seq + tcp_payload_len + flags_count

        # 序列号计算
        flags_count = int(syn_flag) + int(fin_flag)
        next_seq = packet[TCP].seq + tcp_payload_len + flags_count
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        timestamp = packet.time

        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack,
                'length': tcp_payload_len, 'ACK': int(ack_flag), 'SYN': int(syn_flag), 'next_seq': next_seq}
        
        if additional_info == 'PSH':
            PSH_flag = 1
            value['PSH'] = PSH_flag
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        
        if table_num is not None:
            
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            if ack_flag:
                target_values = self.values[table_num][index]
                prior_value, b2b_flag = target_values.process_normal_tcp_element(new_element = value, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    
                    if self.rtt_table:
                        if not b2b_flag:
                            self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'RST', direction=value['direction'])
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'RST', float(rtt*1000), float(timestamp))
                        else:
                            self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, 'Normal', direction=value['direction'])
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'Normal', float(rtt*1000), float(timestamp))
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
    
                
    # 处理普通数据包
    def process_normal_packet(self, packet : TCP, additional_info=None):
        '''
        This function is used to process the normal TCP packet.
        params:
            packet (scapy.Packet): The packet to be processed.
            additional_info (str): The additional information to be processed.
        return:
            None
        function:
            for NetworkTrafficMonitor:
                add_ip_and_record_activity
            for RTTTable:
                add_rtt_sample
        '''
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) != 0
        syn_flag = (tcp_flags & 0x02) != 0
        fin_flag = (tcp_flags & 0x01) != 0
        # 获取IP头部长度和TCP头部长度
        tcp_header_len = packet[TCP].dataofs * 4
        if ":" in src_ip:
            ip_header_len = 40
        else:
            ip_header_len = packet[IP].ihl * 4

    # 计算TCP有效负载长度
        if ":" in src_ip:
            ip_total_length = packet[IPv6].plen
        else:
            ip_total_length = packet[IP].len  # IP总长度包括IP头和TCP段
        # IP数据负载长度 - IP头部长度 - TCP头部长度
        tcp_payload_len = ip_total_length - ip_header_len - tcp_header_len
        # 序列号计算
        flags_count = syn_flag or fin_flag
        next_seq = packet[TCP].seq + tcp_payload_len + flags_count
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        timestamp = packet.time
        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack,
                'length': tcp_payload_len, 'ACK': int(ack_flag), 'SYN': int(syn_flag), 'next_seq': next_seq, 'FIN': int(fin_flag)}
        if additional_info == 'PSH':
            PSH_flag = 1
            value['PSH'] = PSH_flag
        key_temp = {'ip': ip_compare(src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        

        
        table_num, index = self.lookup(key_temp)
        if table_num is None:
            self.insert(key_temp)
        table_num, index = self.lookup(key)
        target_values: ListBuffer
        if table_num is not None:
            if fin_flag:
                self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'FIN', 1, float(timestamp))
            
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            is_valid, packet_type = self.tcp_state[table_num][index].update_state(value)
            self.tcp_state : list[list[TcpState]]
            if not is_valid:
                return
            else:
                if ack_flag:
                    target_values = self.values[table_num][index]
                    prior_value, res = target_values.process_normal_tcp_element(new_element=value, is_add=True, mtu= self.tcp_state[table_num][index].max_length)
                    if prior_value is not None:
                        request_timestamp = prior_value['timestamp']
                        rtt = timestamp - request_timestamp
                        if prior_value['SYN'] == 1:
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'SYN-ACK', float(rtt*1000), float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'SYN-ACK', 1, float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'SYN-ACK', 1, float(timestamp))
                        elif res == 'PSH':
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'PSH', float(rtt*1000), float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'PSH', 1, float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'PSH', 1, float(timestamp))
                        elif res == 'Back-to-Back':
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'Back-to-Back', float(rtt*1000), float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'Back-to-Back', 1, float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'Back-to-Back', 1, float(timestamp))
                            
                        else:
                            return
                            self.net_monitor.add_or_update_ip_with_rtt(src_ip, 'TCP', 'Normal', float(rtt*1000), float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(src_ip, 'TCP', 'Normal', 1, float(timestamp))
                            self.net_monitor.add_ip_and_record_activity(dst_ip, 'TCP', 'Normal', 1, float(timestamp))
                        # 更新RTT表
                        if self.rtt_table:
                                self.rtt_table.add_rtt_sample(src_ip, dst_ip, float(rtt), timestamp, res, direction=value['direction'])
                else:
                    target_values = self.values[table_num][index]
                    condition1 = lambda x, y: False
                    condition2 = lambda x, y: True
                    target_values.process_element(new_element=value, condition1=condition1, condition2=condition2, is_add=True)
    def calculate_rtt(self, value, prior_value):
        weight = 0
        if 'ts_val' in value and 'ts_val' in prior_value:
            rtt = value['timestamp'] - prior_value['timestamp']
            weight = 1
        if 'seq' in value or 'seq' in prior_value:
            weight = 1
        rtt = value['timestamp'] - prior_value['timestamp']
        return float(rtt), weight
    def flush_table(self):
        '''
        This function is used to flush the table. clear all the entries.
        '''
        for table_num in range(3):
            for index in range(self.size):
                if self.tables[table_num][index] is not None:
                    self.delete_entry(table_num, index)