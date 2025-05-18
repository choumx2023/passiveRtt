import ipaddress
import os
import sys
import copy
import typing
import random
from typing import Callable
from decimal import Decimal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import match_keys, protocol_to_int

stage_name = ['stage_1', 'stage_2', 'stage_3']

class ListBuffer:
    '''
    ListBuffer: a class that implements a list-based buffer with a fixed size
    '''
    def __init__(self, size: int, tcp_state: bool = False) -> None:
        '''
        params:
            size (int): The maximum number of elements that the buffer can hold
        '''
        self.size: int = size
        self.buffer: list[dict] = []
        self.count: int = 0
    
    def __state__(self) -> str:
        return f"ListBuffer(size={self.size}, buffer={self.buffer}), count={self.count}"
    
    def add(self, item: dict) -> None:
        '''
        params:
            item: a dictionary containing the item to be added

        to-do:
            为什么要设置len不等于1的条件，因为item的长度为1的话说明这个是个验证可不可以插入，而不是真实的值
        '''
        if len(item) == 1:
            return
        self.buffer.append(item)
        self.count += 1
        if self.count > self.size:
            self.buffer.pop(0)  # 移除最旧的元素以保持缓冲区大小
            self.count -= 1
    def process_element(
        self,
        new_element: dict,
        condition1: Callable[[dict, dict], bool],
        condition2: Callable[[dict, dict], bool],
        is_add: bool
    ) -> dict | None:
        '''
        非TCP的处理函数
        
        This function processes a new element in the buffer and returns the first matched element and ends the processing when the second condition is met. 
        
        parameters:
            new_element: value
            condition1: a function that takes two arguments (existing_item, new_item) and returns a boolean value, indicating whether to remove the existing item
            condition2: a function that takes two arguments (existing_item, new_item) and returns a boolean value, indicating whether to stop processing
            is_add: a boolean value indicating whether to add the new element to the buffer
        return:
            first_matched_value: the first matched value in the buffer
        '''
        
        first_matched_value: dict | None = None
        # 从新到旧遍历缓冲区
        i = len(self.buffer) - 1
        while i >= 0:
            current_element = self.buffer[i]

            if condition2(current_element, new_element):
                break  # 遇到满足条件2的元素，停止处理
            if condition1(current_element, new_element):
                if first_matched_value is None:
                    first_matched_value =  copy.deepcopy(current_element)
                self.buffer.pop(i)  # 移除满足条件1的元素
                self.count -= 1
            i -= 1
        i = i - 1
        while i >= 0:
            self.buffer.pop(i)
            self.count -= 1
            i -= 1
        if is_add:
            self.add(copy.deepcopy(new_element))  # 如果is_add为真，则添加新元素
            #print('add new element', new_element)
        if (first_matched_value is None) or (new_element['timestamp'] - first_matched_value['timestamp'] > 1 - 1e-3):
            return None
        return first_matched_value
    def process_tcp_element(
        self,
        new_element: dict,
        condition1: Callable[[dict, dict], bool],
        condition2: Callable[[dict, dict], bool],
        is_add: bool
    ) -> dict | None:
        '''
        TCP的处理函数 需要增加
        This function processes a new element in the buffer and returns the first matched element and ends the processing when the second condition is met. 
        
        parameters:
            new_element: value
            condition1: a function that takes two arguments (existing_item, new_item) and returns a boolean value, indicating whether to remove the existing item
            condition2: a function that takes two arguments (existing_item, new_item) and returns a boolean value, indicating whether to stop processing
            is_add: a boolean value indicating whether to add the new element to the buffer
        return:
            first_matched_value: the first matched value in the buffer
        '''
        
        '''
        value = {'timestamp': timestamp, 'seq': packet[TCP].seq, 'ack': packet[TCP].ack,
             'length': tcp_payload_len, 'ACK': int(ack_flag), 'SYN': int(syn_flag), 'next_seq': next_seq,'direction' = 'forward'}
        '''
        if 'Retransmission' in new_element and new_element['ts_val'] == -1:
            return None
        if 'Retransmission' in new_element:
            self.add(copy.deepcopy(new_element))
            return None
        first_matched_value: dict | None = None
        # 从新到旧遍历缓冲区
        i = len(self.buffer) - 1
        while i >= 0:
            current_element = self.buffer[i]

            if condition2(current_element, new_element):
                break  # 遇到满足条件2的元素，停止处理
            if condition1(current_element, new_element):
                if first_matched_value is None:
                    first_matched_value =  copy.deepcopy(current_element)
                self.buffer.pop(i)  # 移除满足条件1的元素
                self.count -= 1
            i -= 1

        if is_add:
            self.add(copy.deepcopy(new_element))  # 如果is_add为真，则添加新元素
            #print('add new element', new_element)
        if (first_matched_value is None) or (new_element['timestamp'] - first_matched_value['timestamp'] > 1 - 1e-3):
            return None
        return first_matched_value
    # typical back-to-back TCP packets
    # 1. C->S: seq = x, ack = y, time = 0.0001
    # 2. C->S: seq = x+50, ack = y, time = 0.0002
    # 3. S->C: seq = y, ack = x+50
    def process_normal_tcp_element(
        self,
        new_element: dict,
        is_add: bool,
        mtu: list | None = None,
        gap_time: float = 0.004
    ) -> tuple[dict, str] | None:
        '''
        This function processes a new element in the buffer and returns the first matched element, aiming to detect normal/back-to-back TCP packets 
        
        parameters:
            new_element: value
            is_add: a boolean value indicating whether to add the new element to the buffer
        return:
            first_matched_value: the first matched value in the buffer
        '''
        if mtu is None:
            mtu = [1000, 1000]
        if 'Retransmission' in new_element and new_element['ts_val'] == -1:
            return None
        if 'Retransmission' in new_element:
            self.add(copy.deepcopy(new_element))
            return None
        RETRANS_flag = False
        first_match_value: dict | None = None
        new_ack = new_element['ack']
        # 假设背靠背的TCP包全部是相同的ACK
        current_seq = -1
        i = len(self.buffer) - 1
        count = 0
        PSH_flag = False
        GAP_flag = False
        is_match = False
        big_packet_flag = False
        match_retransmission = False
        ack_len = new_element['ack_length']
        temp_rtt = -1
        # 如果这个重传包没有ts_val，直接返回

        while i >= 0:
            current_element = self.buffer[i]# 考虑之前的包
            print(current_element)
            # 额外的时间戳检查条件
            if new_element['direction'] != current_element['direction']: # 不同方向
                tsval_match = (current_element.get('ts_val', 0) <= new_element.get('ts_ecr', 0))
                if tsval_match:
                    is_match = True
                # 
                if current_element['seq'] > new_element['ack']:
                    i -= 1
                    continue
                if current_element['seq'] + (current_element['FIN'] == 1) + current_element['length'] < new_element['seq_range']:# 过早的数据包
                    self.buffer.pop(i)# 过期的不留
                    self.count -= 1
                    i -= 1
                    continue
                if current_element['length'] == 0 and (current_element['FIN'] == 0 and current_element['SYN'] == 0):
                    self.buffer.pop(i)
                    self.count -= 1
                    i -= 1
                    continue
                elif current_element['FIN'] == 1 or current_element['SYN'] == 1:
                    if current_element['seq'] == new_element['ack'] - 1:
                        first_match_value = copy.deepcopy(current_element)
                        self.buffer.pop(i)
                        self.count -= 1
                        break
                elif (
                    #current_element['ack']  >= new_element['seq_range'] and current_element['ack']  <= new_element['seq']
                    current_element['seq'] + current_element['length'] > new_element['seq_range'] and # current_element报文没有过时
                    current_element['ack'] <= new_element['seq'] and # current_element没有在new_element之后
                    current_element['seq'] < new_element['ack'] and # current_element在new_element之前
                    (is_match or not ('Retransmission' in current_element)) and # 如果时间戳匹配到了 那就正常，如果没有匹配上 那需要保证current不是重传
                    True
                ):
                    if 'Retransmission' in new_element:
                        RETRANS_flag = True
                        return None
                    # 如果是第一个匹配的包
                    if first_match_value is None:
                        if current_element['next_seq'] == new_element['ack']:
                            GAP_flag = False
                        first_match_value = copy.deepcopy(current_element)
                        # 如果第一个匹配的包是合并了多个包的包，直接定义为背靠背
                        current_element['is_matched'] = True
                        count += 1
                        temp_rtt = new_element['timestamp'] - current_element['timestamp']
                        current_seq = first_match_value['seq']
                        if current_element['length'] >= 1400:
                            big_packet_flag = True
                        ack_len -= current_element['length']
                        if ack_len >= 0:
                            self.buffer.pop(i)
                            self.count -= 1
                    else:# 验证是不是有两个及以上的包
                        # 如果匹配上
                        if (
                            #abs(first_match_value['timestamp'] - current_element['timestamp']) < 1e-4 and count and current_element['length'] > 1000
                            #abs(first_match_value['timestamp'] - current_element['timestamp']) < min(max(temp_rtt * 0.05, 1e-3), 0.02) and 
                            ack_len > 0 and current_element['length'] > 0 and current_element['seq'] + current_element['length'] == current_seq and first_match_value['timestamp'] - current_element['timestamp'] < 30 * 1e-3 
                        ):    
                            ack_len -= current_element['length']
                            current_seq = current_element['seq']
                            if ack_len >= 0:
                                count += 1
                                self.buffer.pop(i)
                                self.count -= 1
                            else:
                                current_element['is_matched'] = True
                            if current_element['length'] >= 1400:
                                big_packet_flag = True
                        # 如果超时了，或者有间隔
                        else :
                            self.buffer.pop(i)
                            self.count -= 1
                # 如果没过期了就删掉
                elif (
                    current_element['seq'] + current_element['length'] < new_element['seq_range']
                ):
                    self.buffer.pop(i)
                    self.count -= 1
            if new_element['direction'] == current_element['direction']:#相同方向
                if new_element['ack'] <= current_element['ack'] :
                    break
            i -= 1
        # 如果是可以添加的包，添加
        if is_add:
            self.add(copy.deepcopy(new_element))
        if RETRANS_flag:
            return None
        # 如果没有找到匹配的包或者延迟太大，返回None
        if first_match_value is None:
            return None
        res = "Back-to-Back"
        # 如果存在GAP，返回GAP
        print('****', count)
        if GAP_flag:
            res = "GAP"
        elif count < 2 :
            if big_packet_flag and first_match_value['is_matched'] == False:
                res = "BIG-PACKET"
            else:
                res = "Normal"  
        return first_match_value, res
    def clear(self) -> None:
        '''
        This function clears the buffer
        '''
        self.buffer = []
        self.count = 0             
            
    def print_lb(self) -> None:
        for item in self.buffer:
            print(self.buffer)
    def __str__(self) -> str:
        return f"ListBuffer(size={self.size}, buffer={self.buffer})"
    def __repr__(self) -> str:
        return self.__str__()
    

def random_compare_listbuffer(l1: ListBuffer | dict, l2: ListBuffer) -> bool:
    '''
    This function compares two ListBuffers and returns True if the first ListBuffer is selected, and False otherwise
    
    havent been used
    '''
    # l1 can be ListBuffer or dict
    if isinstance(l1, dict):
        # if l1 is a dict, assign a fixed weight and use l2's last timestamp
        timestamp = l2.buffer[-1]['timestamp'] if l2.buffer else 0.0
        weight1 = 25
    elif isinstance(l1, ListBuffer):
        # both are ListBuffer
        timestamp = max(l1.buffer[-1]['timestamp'] if l1.buffer else 0.0, l2.buffer[-1]['timestamp'] if l2.buffer else 0.0)
        weight1 = calc_listbuffer_weight(l1, timestamp)
    else:
        # fallback, treat as zero
        timestamp = l2.buffer[-1]['timestamp'] if l2.buffer else 0.0
        weight1 = 0
    weight2 = calc_listbuffer_weight(l2, timestamp)
    if weight1 + weight2 == 0:
        return False
    if random.random() < weight1 / (weight1 + weight2):
        return True
    else:
        return False

def calc_listbuffer_weight(l1: ListBuffer, timestamp: float) -> int:
    # 计算ListBuffer的权重, 返回listbuffer中距离timestamp不超过20的元素的个数
    count = 0
    for item in l1.buffer:
        if isinstance(item, dict) and abs(item.get('timestamp', 0.0) - timestamp) <= 20:
            count += 1
    return count

class TcpState:
    def __init__(self) -> None:
        # 初始化TCP状态
        self._reset_state()
    def _reset_state(self) -> None:
        self.forward_range = [-1, -1] # 
        self.backward_range = [-1, -1]
        self.forward_sack_range = [-1, -1]
        self.backward_sack_range = [-1, -1]
        self.time_series = []
        self.max_length = [-1, -1] # forward, backward
        self.throught_output = [0, 0] # forward, backward
        self.valid_throughput = [0, 0] # forward, backward
        self.live_span = [-1, -1] # start, end
        self.init_seq = [-1, -1] # 前向
        self.end_seq = [-1, -1] # 后向
        self.fin_sign = 0
        self.packet_count = [0, 0]
        # 这里应该加一个更新的操作
        # live_span, throught output  ip地址，端口号
        # return live_span, throught_output, valid_throughput
    def clear(self) -> None:
        self._reset_state()
    def update_state(self, value: dict) -> tuple[bool, str, int, int]:
        '''
        Updates the internal state of the TCP connection based on the provided packet details.

        Arguments:
        value (dict): Dictionary containing details about the TCP packet.
        '''

        # 识别方向
        direction_idx = 0 if value['direction'] == 'forward' else 1
        opp_direction_idx = 1 - direction_idx

        # 更新吞吐量，包计数和活动时间
        self.throught_output[direction_idx] += value['length']
        self.packet_count[direction_idx] += 1
        current_timestamp = value['timestamp']
        if self.live_span[0] == -1:
            self.live_span[0] = current_timestamp
        self.live_span[1] = max(self.live_span[1], current_timestamp)

        # Update max length
        self.max_length[direction_idx] = max(self.max_length[direction_idx], min(value['length'], 1448))

        # Update sequence numbers
        if self.init_seq[direction_idx] == -1:
            self.init_seq[direction_idx] = value['seq']
        self.end_seq[direction_idx] = max(self.end_seq[direction_idx], value['next_seq'])

        # Update forward and backward ranges
        next_seq = value['next_seq']
        ack_length = 0
        ack = value['ack']
        update_forward = next_seq > self.forward_range[direction_idx]
        update_backward = ack > self.backward_range[opp_direction_idx]
        print('-------------------')
        print(value)
        print(self.forward_range, self.backward_range)
        seq_range = self.backward_range[opp_direction_idx]
        Heartbeat = False
        if next_seq == self.forward_range[direction_idx] - 1:
            Heartbeat = True
        if update_forward:
            self.forward_range[direction_idx] = next_seq
            self.valid_throughput[direction_idx] += value['length']

        if update_backward:
            ack_length = ack - self.backward_range[opp_direction_idx]
            self.backward_range[opp_direction_idx] = ack
            
        print('****',self.forward_range, self.backward_range)
        # Determine packet type and validity
        is_valid, packet_type = self._classify_packet(value, update_forward, update_backward, is_heartbeat=Heartbeat)
        
        return is_valid, packet_type, ack_length, seq_range

    def _classify_packet(
        self,
        value: dict,
        update_forward: bool,
        update_backward: bool,
        is_heartbeat: bool = False
    ) -> tuple[bool, str]:
        '''
        Classifies the type of the packet based on TCP flags and updates.
        '''
        if value.get('RST', False):
            return False, 'Reset'
        if value.get('SYN', False) and value.get('ACK', False):
            if update_forward:
                return True, 'SYN-ACK'
            else:
                return False, 'Retransmission'
        if value.get('SYN', False):
            if update_forward and update_backward:
                return True, 'SYN'
            else:
                return False, 'Retransmission'
        if value.get('FIN', False):
            if update_forward:
                return True, 'FIN'
            else:
                return False, 'Retransmission'
        if value['length'] > 0 and not update_backward and not update_forward:
            return False, 'Retransmission'
        if value['length'] == 0:
            if is_heartbeat:
                return True, 'Heartbeat'
            elif update_backward:
                return True, 'Pure ACK'
            else:
                return False, 'Duplicate ACK'
        return True, 'Normal'
    def __update_state(self, value: dict) -> tuple[bool, str]:
        '''
        value = {
            'timestamp': packet.time,
            'seq': packet[TCP].seq, 
            'ack': packet[TCP].ack, 
            'SYN':1, 
            'ACK': ack_flag,
            'FIN' : fin_flag, 
            'RST' : rst_flag,
            'PSH' : psh_flag,  
            'length': tcp_payload_len, 
            'next_seq': next_seq, 'direction': 'forward'
            }

        return
            valid: a boolean value indicating whether the packet is valid
            type: a string indicating the type of the packet
        '''
        judge = 0
        is_valid, packet_type = True, None
        # 更新live_span, throught_output, max_length
        if value['direction'] == 'forward':
            self.max_length[0] = max(self.max_length[0], min(value['length'], 1448))
            self.throught_output[0] += max(value['length'], 0)
            if self.live_span[0] == -1:
                self.live_span[0] = value['timestamp']
            self.live_span[1] = max(self.live_span[1], value['timestamp'])
        else:
            self.max_length[1] = max(self.max_length[1], min(value['length'],1448))
            self.throught_output[1] += max(value['length'], 0)
            if self.live_span[0] == -1:
                self.live_span[0] = value['timestamp']
            self.live_span[1] = max(self.live_span[1], value['timestamp'])
            
        # 更新forward_range, backward_range, valid_throughput
        if value['direction'] == 'forward':
            self.packet_count[0] += 1
            if self.init_seq[0] == -1:
                self.init_seq[0] = value['seq']
            if self.end_seq[0] <= value['next_seq']:
                self.end_seq[0] = value['next_seq']
            if value['next_seq'] > self.forward_range[1]:
                self.forward_range[1] = value['next_seq']
                judge += 1
                if self.forward_range[1] != -1:
                    self.valid_throughput[0] += value['length']
                else:
                    self.valid_throughput[0] = value['length']
            if value['ack'] > self.backward_range[0]:
                self.backward_range[0] = value['ack']
                judge += 2

            # 如果没有更新，判断是不是重传
            if judge == 0:
                if value['SYN'] and value['ACK']:
                    packet_type = 'SYN-ACK'
                elif value['SYN']:
                    packet_type = 'SYN'
                elif value['length'] == 0:
                    packet_type, is_valid= 'Heartbeat', False
                else:
                    packet_type, is_valid = 'Retransmission', False
            elif judge == 1:
                packet_type = 'Back-to-Back'
            elif judge == 2:
                packet_type = 'Pure ACK'
            else:
                packet_type = 'Normal'
        else:
            self.packet_count[1] += 1
            if self.init_seq[1] == -1:
                self.init_seq[1] = value['seq']
            if self.end_seq[1] <= value['next_seq']:
                self.end_seq[1] = value['next_seq']
            if value['next_seq'] > self.backward_range[1]:
                self.backward_range[1] = value['next_seq']
                judge += 1
                if self.backward_range[1] != -1:
                    self.valid_throughput[1] += value['length']
                else:
                    self.valid_throughput[1] = value['length']
            if value['ack'] > self.forward_range[0]:
                self.forward_range[0] = value['ack']
                judge += 2

            if judge == 0:
                if value['SYN'] and value['ACK']:
                    packet_type = 'SYN-ACK'
                elif value['SYN']:
                    packet_type = 'SYN'
                elif value['length'] == 0:
                    packet_type, is_valid = 'Heartbeat', False
                else:
                    packet_type, is_valid = 'Retransmission', False
            elif judge == 1:
                packet_type = 'Back-to-Back candidate'
            elif judge == 2:
                packet_type = 'Pure ACK'
            else:
                packet_type = 'Normal'
        return is_valid, packet_type
    
    def get_flow_record(self) -> dict:
        '''
        This function returns the flow record of the TCP connection
        params:
            None
        return:
            flow_record: a dictionary containing the flow record of the TCP connection
        '''
        return {
            'live_span': [float(self.live_span[0]) , float(self.live_span[1])],
            'live_time': float(self.live_span[1]) - float(self.live_span[0]),
            'throught_output': self.throught_output,
            'valid_throughput': self.valid_throughput,
            'max_length': self.max_length,
            'fin_sign': self.fin_sign,
            'total_throughput': [self.end_seq[0] - self.init_seq[0], self.end_seq[1] - self.init_seq[1]],
            'live_time': self.live_span[1] - self.live_span[0],
            'all_output' : self.valid_throughput[0] + self.valid_throughput[1],
            'packet_count': self.packet_count,
            'average_packet_length': [1.0 * self.throught_output[0] / (1e-3 + self.packet_count[0]), 1.0 * self.throught_output[1] /(1e-3 +  self.packet_count[1])]
        }
    def __str__(self) -> str:
        return f"TcpState(forward_range={self.forward_range}, backward_range={self.backward_range})"

class CuckooHashTable:
    '''
    key: a dictionary containing the keys to be inserted,
        essential: 'ip' and 'protocol', ip is a tuple containing source IP and destination IP
        optional : 'port', a tuple containing source port and destination port
    value: a dictionary containing the values to be inserted
        essential : timestamp
        optional : 
            for ICMP: 'type', 'code'
            for TCP : 'src_port', 'dst_port', 'seq_num', 'ack_num', 'syn', 'ack'
            for NTP : 'code'
            for DNS : 'query', 'response', 'id'
    '''
    def __init__(
        self,
        initial_size: int = 100013,
        buffersize: int = 300,
        type: str = 'Normal'
    ) -> None:
        '''
        Initialize the CuckooHashTable with the given initial size and buffer size
        parameter:
            initial_size: the initial size of the hash table
            buffersize: the size of the buffer
        我觉得最大30就够了
        '''
        self.size: int = initial_size
        self.buffersize: int = buffersize
        self.type: str = type
        self.tables: list[list[list | tuple[dict, str] | None]] = [[None] * self.size for _ in range(3)]  # [None] or [key, stage_name] (as tuple)
        self.values: list[list[ListBuffer]] = [[ListBuffer(buffersize) for _ in range(self.size)] for _ in range(3)]
        if self.type == 'TCP':
            self.tcp_state: list[list[TcpState]] = [[TcpState() for _ in range(self.size)] for _ in range(3)]
        self.num_items: int = 0
        self.rehash_threshold: float = 0.6
        self.max_rehash_attempts: int = 5
        
    def hash_ip(self, ip : ipaddress.IPv4Address | ipaddress.IPv6Address) -> int:
        '''
        This function hashes an IP address and returns the hash value
        params:
            ip: an IPv4 or IPv6 address
        return:
            ip_int: the hash value of the IP address
        '''
        ip_int = int(ipaddress.ip_address(ip))
        if ipaddress.ip_address(ip).version == 6:
            ip_int = ip_int % 2**64  # 简化 IPv6 地址
        return ip_int 
    def hash_functions(self, key_dict: dict, function_id: int) -> int:
        '''
        This function hashes the given key dictionary using the specified hash function
        params:
            key_dict: a dictionary containing the keys to be hashed
            function_id: the ID of the hash function to be used
        return:
            hash_value: the hash value of the key dictionary
        '''
        result = 0
        if 'ip' in key_dict and 'protocol' in key_dict:
            ip_hashes = [int(ipaddress.ip_address(ip)) & ((1 << 64) - 1) for ip in key_dict['ip']]
            # 对IP哈希值求和并加上它们的乘积，以确保顺序不影响最终哈希值
            result += sum(ip_hashes) + (ip_hashes[0] % self.size) * (ip_hashes[1] % self.size)

            result += protocol_to_int(key_dict['protocol'])

        if 'port' in key_dict:
            port_hashes = [(port << 16) for port in key_dict['port']]
            # 类似地处理端口号，保证顺序不影响哈希值
            result += sum(port_hashes) + (port_hashes[0] % self.size) * (port_hashes[1] % self.size)
        else :
            result *= 19
            result += 13
        if function_id == 0:
            return result % self.size
        elif function_id == 1:
            return (result * 2654435761 % 2**32) % self.size
        else:  # function_id == 2
            return (result * 805306457 % 2**32) % self.size
    def _rehash(self) -> None:
        '''
        This function rehashes the hash table when collisions occur, doubling the size of the hash table and reinserting all items
        '''
        old_size: int = self.size
        self.size *= 2
        old_tables = self.tables
        old_values = self.values
        self.tables = [[None] * self.size for _ in range(3)]
        self.values = [
            [
                ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                for _ in range(self.size)
            ]
            for _ in range(3)
        ]
        self.num_items = 0  # Reset item count

        for table in range(3):
            for i in range(old_size):
                key_entry = old_tables[table][i]
                # Accept both list and tuple for key_entry, but ensure it's a tuple[dict, str]
                if (
                    key_entry is not None
                    and (isinstance(key_entry, (list, tuple)))
                    and len(key_entry) == 2
                    and isinstance(key_entry[0], dict)
                    and isinstance(key_entry[1], str)
                ):
                    # Convert to tuple if not already
                    if not isinstance(key_entry, tuple):
                        key_entry_tuple = (key_entry[0], key_entry[1])
                    else:
                        key_entry_tuple = key_entry
                    self._insert_directly(key_entry_tuple, old_values[table][i])
        '''
        old_values1 = self.values1
        old_values2 = self.values2
        old_index1 = self.index1
        old_index2 = self.index2
        self.size *= 2
        self.index1 = [None] * self.size
        self.index2 = [None] * self.size
        self.values1 = [ListBuffer(self.buffer_size) for _ in range(self.size)]
        self.values2 = [ListBuffer(self.buffer_size) for _ in range(self.size)]
        self.num_items = 0

        for i in range(len(old_index1)):
            if old_index1[i] is not None:
                self._insert_directly(old_index1[i])
                table_num, index = self.lookup(old_index1[i])
                if table_num is not None:
                    target_values = self.values1 if table_num == 1 else self.values2
                    for item in old_values1[i].buffer:
                        target_values[index].add(item)
            if old_index2[i] is not None:
                self._insert_directly(old_index2[i])
                table_num, index = self.lookup(old_index2[i])
                if table_num is not None:
                    target_values = self.values1 if table_num == 1 else self.values2
                    for item in old_values2[i].buffer:
                        target_values[index].add(item)
        '''
    def insert(self, key: dict, value: dict | None = None) -> bool:
        '''
        function: insert a key into the index table , if the insertion fails after multiple attempts and rehashing, return False
        parameter:
            key: a dictionary containing the keys to be inserted
            value: a dictionary containing the values to be inserted
        return:
            success: a boolean value indicating whether the insertion is successful
        '''
        if value is None:
            value = {}
        table_num, _ = self.lookup(key)
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            # Defensive: check structure before indexing
            table_entry = self.tables[table_num][index]
            if (
                isinstance(table_entry, list)
                and len(table_entry) == 2
                and isinstance(table_entry[0], dict)
                and isinstance(table_entry[1], str)
            ):
                target_key = table_entry[0]
                if target_key.get('ip') == key.get('ip') and target_key.get('protocol') == key.get('protocol'):
                    value['direction'] = 'forward'
                else:
                    value['direction'] = 'backward'
                self.values[table_num][index].add(value)
                return True
            else:
                # Should not happen, but skip if structure is not as expected
                return False

        generate_key: tuple[dict, str] = (key, 'stage_1')
        if value == {}:
            value['direction'] = 'forward'
        success, _ = self._insert_directly(generate_key, value)

        if not success:
            self._rehash()
            success, _ = self._insert_directly(generate_key, value)
        return success

    def _insert_directly(
        self,
        key: tuple[dict, str],
        value: dict | ListBuffer
    ) -> tuple[bool, list]:
        '''
        function: insert a key into the index table without rehashing
        parameter:
            key: a list [dict, str] containing the keys to be inserted
            value: a dict or ListBuffer containing the values to be inserted
        return:
            is_success: a boolean value indicating whether the insertion is successful
            path: a list containing the path of the key
        '''
        path: list = []
        current_key = key
        current_value = value
        starting_table = 0
        is_end, is_increase = False, True
        is_Lb = False
        while True:
            for table_id in range(starting_table, 3):
                # Ensure current_key is a tuple[dict, str]
                if (
                    isinstance(current_key, (list, tuple))
                    and len(current_key) == 2
                    and isinstance(current_key[0], dict)
                    and isinstance(current_key[1], str)
                ):
                    # If it's a list, convert to tuple for consistency
                    if not isinstance(current_key, tuple):
                        current_key = (current_key[0], current_key[1])
                    index = self.hash_functions(current_key[0], table_id)
                else:
                    continue  # skip illegal structure
                if not is_Lb:
                    # current_value is dict
                    if self.tables[table_id][index] is None:
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        if isinstance(current_value, dict):
                            self.values[table_id][index].add(current_value)
                        is_Lb = True
                        self.num_items += 1
                        is_end = True
                        break
                    else:
                        evicted_key = copy.deepcopy(self.tables[table_id][index])
                        # Always convert to tuple for consistency
                        if not isinstance(evicted_key, tuple) and isinstance(evicted_key, (list, tuple)) and len(evicted_key) == 2:
                            evicted_key = (evicted_key[0], evicted_key[1])
                        evicted_value = copy.deepcopy(self.values[table_id][index])
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        if isinstance(current_value, dict):
                            self.values[table_id][index].add(current_value)
                        current_key = evicted_key
                        path.append(current_key)
                        if (
                            isinstance(evicted_key, tuple)
                            and len(evicted_key) == 2
                            and isinstance(evicted_key[1], str)
                            and evicted_key[1] == 'stage_3'
                        ):
                            current_value = evicted_value
                            is_end = True
                            is_increase = False
                            break
                        elif (
                            isinstance(evicted_key, tuple)
                            and len(evicted_key) == 2
                            and isinstance(evicted_key[1], str)
                        ):
                            # promote stage
                            evicted_key = (evicted_key[0], stage_name[stage_name.index(evicted_key[1]) + 1])
                            current_key = evicted_key
                            current_value = evicted_value
                            is_Lb = True
                else:
                    # current_value is ListBuffer
                    if current_key in path:
                        is_end = True
                        break
                    if self.tables[table_id][index] is None:
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        if isinstance(current_value, ListBuffer):
                            for element in current_value.buffer:
                                self.values[table_id][index].add(element)
                        is_Lb = True
                        self.num_items += 1
                        is_end = True
                        break
                    else:
                        evicted_key = copy.deepcopy(self.tables[table_id][index])
                        if not isinstance(evicted_key, tuple) and isinstance(evicted_key, (list, tuple)) and len(evicted_key) == 2:
                            evicted_key = (evicted_key[0], evicted_key[1])
                        evicted_value = copy.deepcopy(self.values[table_id][index])
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        if isinstance(evicted_value, ListBuffer):
                            for element in evicted_value.buffer:
                                self.values[table_id][index].add(element)
                        current_key = evicted_key
                        path.append(current_key)
                        if (
                            isinstance(evicted_key, tuple)
                            and len(evicted_key) == 2
                            and isinstance(evicted_key[1], str)
                            and evicted_key[1] == 'stage_3'
                        ):
                            current_value = evicted_value
                            is_end = True
                            is_increase = False
                            break
                        elif (
                            isinstance(evicted_key, tuple)
                            and len(evicted_key) == 2
                            and isinstance(evicted_key[1], str)
                        ):
                            evicted_key = (evicted_key[0], stage_name[stage_name.index(evicted_key[1]) + 1])
                            current_key = evicted_key
                            current_value = evicted_value
            if is_end:
                break
        return key not in path, path

    def lookup(self, key: dict) -> tuple[int, int] | tuple[None, None]:
        '''
            key: a dictionary containing the keys to be looked up
        return:
            (table_num, index): the table number and index where the key is found, or (None, None) if not found
        '''
        for table_id in range(3):
            index = self.hash_functions(key, table_id)
            table_entry = self.tables[table_id][index]
            if (
                isinstance(table_entry, (list, tuple))
                and len(table_entry) == 2
                and isinstance(table_entry[0], dict)
                and isinstance(table_entry[1], str)
                and match_keys(table_entry[0], key)
            ):
                return table_id, index
        return None, None  # Key not found
    def delete(self, key : dict) -> bool:
        '''
        parameter :
            key: a dictionary containing the keys to be deleted
        return :
            removed: a boolean value indicating whether the deletion is successful
        '''
        for table_id in range(3):
            index = self.hash_functions(key, table_id)
            table_entry = self.tables[table_id][index]
            if (
                table_entry is not None
                and isinstance(table_entry, (list, tuple))
                and len(table_entry) == 2
                and isinstance(table_entry[0], dict)
                and match_keys(table_entry[0], key)
            ):
                self.tables[table_id][index] = None
                self.num_items -= 1
                return True
        return False  # Key not found

    def flush_tables(self) -> None:
        '''
        This function clears the hash table
        '''
        is_tcp = self.type == 'TCP'
        if not is_tcp:
            return
        for table_id in range(3):
            for i in range(self.size):
                if self.tables[table_id][i] is not None:
                    self.values[table_id][i].clear()
                    self.tcp_state[table_id][i].clear()
                    
                
    def print_tables(self) -> None:
        for table_id in range(3):
            print(f"Table {table_id}:")
            for i in range(self.size):
                if self.tables[table_id][i] is not None:
                    print(f"Index {i}: {self.tables[table_id][i]} -> {self.values[table_id][i]}")
                    print()
            
        
    def stats(self) -> None:
        print(f"Current size: {self.size}")
        print(f"Number of items: {self.num_items}")
        print(f"Load factor: {self.num_items / (2 * self.size)}")


    
    
    def __str__(self) -> str:
        return f"CuckooHashTable(size={self.size}, num_items={self.num_items})"
    
     