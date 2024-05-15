from scapy.all import IP, ICMP, UDP, DNS, NTP, TCP
from .CuckooHashTable import CuckooHashTable, ListBuffer
from .RttTable import RTTTable
from utils.utils import extract_ip
import time
# CuckooHashTable key要求是字典
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
    def __init__(self, initial_size=10013):
        super().__init__(initial_size, buffersize = 50)
        self. rtt_table = RTTTable()
        self.size : int
        self.buffersize : int
        self.tables : list[list[bool, str]]
        self.values : list[list[ListBuffer]]
    def add_packet(self, packet):
        if ICMP in packet:
            self.process_icmp_packet(packet)
        elif DNS in packet and UDP in packet:
            self.process_dns_packet(packet)
        elif NTP in packet:
            self.process_ntp_packet(packet)
        else:
            pass

    def process_icmp_packet(self, packet):
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
            if icmp_type == 8:  # ICMP请求
                # 直接插入，等待响应
                target_values = self.values[table_num][index]
                
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element( new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif icmp_type == 0:  # ICMP响应
                # 尝试找到匹配的请求
                print(key)
                print(value)
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: x['type'] == 8 and x['seq'] == y['seq']
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    print(f"prior_value: {prior_value}")
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'ICMP')
                        print(f"RTT: {rtt}")
                # 可选择删除请求记录或保留以支持多次测量
                # self.delete(key)

    def process_dns_packet(self, packet):
        src_ip, dst_ip = extract_ip(packet)
        dst_port = packet[UDP].dport
        src_port = packet[UDP].sport
        dns_id = packet[DNS].id
        qr = packet[DNS].qr

        timestamp = packet.time
        key = {'ip':(src_ip, dst_ip),'port':(src_port, dst_port), 'protocol': 'DNS'}
        value = {'timestamp': timestamp, 'dns_id': dns_id, 'is_qr': qr}
        table_num, index = self.lookup(key)
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            if qr == 0:  # Query
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif qr == 1:  # Response
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['dns_id'] == new_item['dns_id'] and existing_item['is_qr'] == 0 and new_item['is_qr'] == 1
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    # find the matching request and calculate RTT
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp          
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'DNS')
                        print(f"RTT: {rtt}")
                    #self.delete(request_key)                    

    def process_ntp_packet(self, packet: NTP):
        src_ip, dst_ip = extract_ip(packet)
        dst_port = packet[UDP].dport
        src_port = packet[UDP].sport
        ntp_mode = packet[NTP].mode
        timestamp = packet.time
        orig_timestamp = packet[NTP].orig
        recv_timestamp = packet[NTP].recv
        sent_timestamp = packet[NTP].sent
        ref_timestamp = packet[NTP].ref
        key = {'ip':(src_ip, dst_ip),'port':(src_port, dst_port), 'protocol': 'NTP'}
        value = {'timestamp': timestamp, 'mode': ntp_mode, 'orig': orig_timestamp, 'ref': ref_timestamp, 'recv': recv_timestamp, 'sent': sent_timestamp}
        table_num, index = self.lookup(key)
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            if ntp_mode == 3:
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
            elif ntp_mode == 4:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['mode'] == 3 and new_item['mode'] == 4 and new_item['ref'] == existing_item['orig']
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'NTP')
                        print(f"RTT: {rtt}")
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
    def __init__(self, initial_size=1019, rtt_table=None):
        super().__init__(initial_size)
        self.rtt_table = rtt_table or RTTTable()  # RTT表实例，用于存储RTT计算结果

    def add_packet(self, packet):
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            syn_flag = (tcp_flags & 0x2) >> 1
            ts_val, _ = self.extract_tcp_options(packet)
            if syn_flag:
                self.process_syn_packet(packet)
            elif ts_val is not None:
                self.process_timestamp_packet(packet)
            else:
                self.process_normal_packet(packet)
            # 尝试找到匹配的数据包并计算RTT
            
    def extract_tcp_options(self, packet):
        """从TCP数据包中提取选项，特别是Timestamp"""
        ts_val, ts_ecr = None, None
        if TCP in packet:
            for option in packet[TCP].options:
                if option[0] == 'Timestamp':
                    ts_val, ts_ecr = option[1]
                    break
        return ts_val, ts_ecr
    # 处理SYN数据包
    def process_syn_packet(self, packet):
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) >> 4
        timestamp = packet.time
        key ={'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        value = {'timestamp': packet.time, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack, 'SYN':1, 'ACK': ack_flag}
        table_num, index = self.lookup(key)
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            # ACK & SYN
            if ack_flag:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item['SYN'] == 1 and existing_item['ACK'] == 0 and existing_item['seq'] == new_item['ack'] - 1
                condition2 = lambda existing_item, new_item: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'SYN')
            # SYN
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
    # 处理Timestamp数据包
    def process_timestamp_packet(self, packet):
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) >> 4
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        ts_val, ts_ecr = self.extract_tcp_options(packet)
        timestamp = packet.time
        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': ack_flag , 'ts_val': ts_val, 'ts_ecr': ts_ecr}
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        if table_num is not None:
            if ack_flag:
                target_values = self.values[table_num][index]
                condition1 = lambda existing_item, new_item: existing_item.get('ts_val') is not None and existing_item.get('ts_val') == new_item.get('ts_val') and existing_item.get('ts_ecr') == new_item.get('ts_ecr')
                condition2 = lambda x, y: False
                prior_value = target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = False)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'Timestamp')
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
    # 处理普通数据包
    def process_normal_packet(self, packet):
        src_ip, dst_ip = extract_ip(packet)
        tcp_flags = packet[TCP].flags
        ack_flag = (tcp_flags & 0x10) >> 4
        key = {'ip': (src_ip, dst_ip), 'protocol': 'TCP', 'port': (packet[TCP].sport, packet[TCP].dport)}
        timestamp = packet.time
        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack, 'length': len(packet), 'ACK': ack_flag, 'SYN': 0}
        table_num, index = self.lookup(key)
        if table_num is None:
            self.insert(key)
        table_num, index = self.lookup(key)
        target_values : ListBuffer
        if table_num is not None:
            if ack_flag:
                target_values = self.values[table_num][index]
                prior_value = target_values.process_normal_tcp_element(new_element = value, is_add = True)
                if prior_value is not None:
                    request_timestamp = prior_value['timestamp']
                    rtt = timestamp - request_timestamp
                    # 更新RTT表
                    if self.rtt_table:
                        self.rtt_table.add_rtt_sample(src_ip, dst_ip, rtt, timestamp, 'Normal')
            else:   
                target_values = self.values[table_num][index]
                condition1 = lambda x, y: False
                condition2 = lambda x, y: True
                target_values.process_element(new_element = value, condition1 = condition1, condition2 = condition2, is_add = True)
    def calculate_rtt(self, value, prior_value):
        weight = 0
        if 'ts_val' in value and 'ts_val' in prior_value:
            rtt = value['timestamp'] - prior_value['timestamp']
            weight = 1
        if 'seq' in value or 'seq' in prior_value:
            weight = 1
        rtt = value['timestamp'] - prior_value['timestamp']
        return rtt, weight
