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
    def __init__(self, size : int, tcp_state = False) -> None:
        '''
        params:
            size (int): The maximum number of elements that the buffer can hold
        '''
        self.size = size
        self.buffer = []
        self.count = 0
        # 如果是TCP，需要记录TCP状态
    def add(self, item : dict) -> None:
        '''
        params:
            item: a dictionary containing the item to be added

        to-do:
            为什么要设置len不等于1的条件
        '''
        if len(item) == 1:
            return
        self.buffer.append(item)
        self.count += 1
        if self.count > self.size:
            self.buffer.pop(0)  # 移除最旧的元素以保持缓冲区大小

    def process_element(self, new_element: list, condition1 : Callable[[dict, dict], bool], condition2 : Callable[[dict, dict], bool], is_add : bool) -> list:
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
        
        first_matched_value = None
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
            i -= 1

        if is_add:
            self.add(copy.deepcopy(new_element))  # 如果is_add为真，则添加新元素
            #print('add new element', new_element)
        if (first_matched_value is None) or (new_element['timestamp'] - first_matched_value['timestamp'] > 1 - 1e-3):
            return None
        
        return first_matched_value
    def process_tcp_element(self, new_element: list, condition1 : Callable[[dict, dict], bool], condition2 : Callable[[dict, dict], bool], is_add : bool) -> list:
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
        self.tcp_state : dict
        first_matched_value = None
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
    def process_normal_tcp_element(self, new_element : dict ,is_add : bool, mtu : list = [1000, 1000]) -> typing.Union[list, bool]:
        '''
        This function processes a new element in the buffer and returns the first matched element, aiming to detect normal/back-to-back TCP packets 
        
        parameters:
            new_element: value
            is_add: a boolean value indicating whether to add the new element to the buffer
        return:
            first_matched_value: the first matched value in the buffer
        '''
        self.tcp_state : dict
        first_match_value = None
        new_ack = new_element['ack']
        # 假设背靠背的TCP包全部是相同的ACK
        old_ack = -1
        i = len(self.buffer) - 1
        count = 0
        PSH_flag = False
        GAP_flag = True
        while i >= 0:
            current_element = self.buffer[i]# 考虑之前的包
            maxium_length = -1
            if new_element['direction'] != current_element['direction']: # 不同方向
                if current_element['ack'] < new_element['seq']:# 过早的数据包
                    self.buffer.pop(i)# 过期的不留
                elif current_element['ack'] == new_element['seq'] and current_element['seq'] <= new_element['ack'] :
                    if 'PSH' in current_element and current_element['PSH'] == 1:
                        PSH_flag = True
                    if first_match_value is None:
                        if current_element['next_seq'] == new_element['ack']:
                            GAP_flag = False
                        first_match_value = copy.deepcopy(current_element)
                        maxium_length = first_match_value['length']
                        count += 1
                    else:# 验证是不是有两个及以上的包
                        # 如果不止发送了一个packet，要保留两个
                        if abs(first_match_value['timestamp'] - current_element['timestamp']) < 1e-4 and count and current_element['length'] >= mtu[new_element['direction'] == 'forward'] and current_element['length'] > 1000:
                            maxium_length = current_element['length']
                            count += 1
                            self.buffer.pop(i)
                        # 如果连续发送不少于三个，删掉前n-2个
                        else :
                            self.buffer.pop(i)
            if new_element['direction'] == current_element['direction']:#相同方向
                if current_element['ack'] == new_ack:
                    count = 0
                    break
            i -= 1
        if is_add:
            self.add(copy.deepcopy(new_element))
        if first_match_value is None or new_element['timestamp'] - first_match_value['timestamp'] > 1 - 1e-3:
            return None, None
        if not PSH_flag:# 没有psh。也没有back-to-back
            return None, None
        res = "Back-to-Back"
        if count < 2 and PSH_flag:
            res = "PSH"
        if GAP_flag:
            res = "GAP"
        return first_match_value, res
    def clear(self) -> None:
        '''
        This function clears the buffer
        '''
        self.buffer = []
        self.count = 0             
            
    def print_lb(self):
        for item in self.buffer:
            print(self.buffer)
    def __str__(self) -> str:
        return f"ListBuffer(size={self.size}, buffer={self.buffer})"
    def __repr__(self) -> str:
        return self.__str__()
    
    

def random_compare_listbuffer(l1 : ListBuffer | dict, l2 : ListBuffer) -> bool:
    '''
    This function compares two ListBuffers and returns True if the first ListBuffer is selected, and False otherwise
    
    havent been used
    '''
    timestamp = max(l1.buffer[-1]['timestamp'], l2.buffer[-1]['timestamp'])
    if isinstance(l1, dict):
        weight1 = 25
    else:
        weight1 = calc_listbuffer_weight(l1, timestamp)
    weight2 = calc_listbuffer_weight(l2, timestamp)
    if random.random() < weight1 / (weight1 + weight2):
        return True
    else:
        return False
def calc_listbuffer_weight(l1, timestamp) -> int:
    # 计算ListBuffer的权重, 返回listbuffer中距离timestamp不超过20的元素的个数
    count = 0
    for i in range(len(l1)):
        if abs(l1[i] - timestamp) <= 20:
            count += 1
    return count

class TcpState():
    def __init__(self) -> None:
        self.forward_range = [-1, -1]
        self.backward_range = [-1, -1]
        self.forward_sack_range = [-1, -1]
        self.backward_sack_range = [-1, -1]
        self.time_series = []
        self.max_length = [-1, -1]
        self.throught_output = [0, 0]
        self.valid_throughput = [0, 0]
        self.init_seq = [-1, -1]
        self.end_seq = [-1, -1]
        self.live_span = [-1, -1]
        self.fin_sign = 0
        self.packet_count = [0, 0]
    def clear(self) -> None:
        self.forward_range = [-1, -1]
        self.backward_range = [-1, -1]
        self.forward_sack_range = [-1, -1]
        self.backward_sack_range = [-1, -1]
        self.time_series = []
        self.max_length = [-1, -1] # forward, backward
        self.throught_output = [0, 0] # forward, backward
        self.valid_throughput = [0, 0] # forward, backward
        self.live_span = [-1, -1] # start, end
        self.init_seq = [-1, -1]
        self.end_seq = [-1, -1]
        self.fin_sign = 0
        self.packet_count = [0, 0]
        # 这里应该加一个更新的操作
        # live_span, throught output  ip地址，端口号
        # return live_span, throught_output, valid_throughput
    def update_state(self, value : dict) -> None:
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
            if 'FIN' in value and value['FIN']:
                self.fin_sign |=1
            self.max_length[0] = max(self.max_length[0], value['length'])
            self.throught_output[0] += value['length']
            if self.live_span[0] == -1:
                self.live_span[0] = value['timestamp']
            self.live_span[1] = max(self.live_span[1], value['timestamp'])
        else:
            if 'FIN' in value and value['FIN']:
                self.fin_sign |= 2
            self.max_length[1] = max(self.max_length[1], value['length'])
            self.throught_output[1] += value['length']
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

class CuckooHashTable():
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
    def __init__(self, initial_size = 100013, buffersize = 30, type = 'Normal') -> None:
        '''
        Initialize the CuckooHashTable with the given initial size and buffer size
        parameter:
            initial_size: the initial size of the hash table
            buffersize: the size of the buffer
        
        
        我觉得最大30就够了
        '''
        self.size = initial_size
        self.buffersize = buffersize
        self.type = type
        self.tables = [[None] * self.size for _ in range(3)]# [None] or [key, stage_name]
        self.values = [[ListBuffer(buffersize) for _ in range(self.size)] for _ in range(3)]
        if self.type == 'TCP':
            self.tcp_state =[ [ TcpState() for _ in range(self.size)] for _ in range(3)]
        self.num_items = 0
        self.rehash_threshold = 0.6
        self.max_rehash_attempts = 5
        
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

        if function_id == 0:
            return result % self.size
        elif function_id == 1:
            return (result * 2654435761 % 2**32) % self.size
        else:  # function_id == 2
            return (result * 805306457 % 2**32) % self.size
    def _rehash(self) -> None:
        '''
        This function rehashes the hash table when collisions occur, doubling the size of the hash table and reinserting all items
        params:
            None
        return:
            None
        '''
        # 重新哈希的实现
        old_size = self.size
        self.size *= 2
        old_tables = self.tables
        old_values = self.values
        self.tables = [[None] * self.size for _ in range(3)]
        self.values = [[ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize) for _ in range(self.size)] for _ in range(3)]
        self.num_items = 0  # Reset item count

        for table in range(3):
            for i in range(old_size):
                if old_tables[table][i] is not None:
                    self._insert_directly(old_tables[table][i], old_values[table][i])
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
    def insert(self, key: dict, value  : dict = {}) -> bool:
        '''
        function: insert a key into the index table , if the insertion fails after multiple attempts and rehashing, return False
        parameter:
            key: a dictionary containing the keys to be inserted
            value: a ListBuffer containing the values to be inserted
        return:
            success: a boolean value indicating whether the insertion is successful
        '''
        table_num, _= self.lookup(key)
        if table_num is not None:
            index = self.hash_functions(key, table_num)
            target_key = self.tables[table_num][index][0]
            if target_key['ip'] == key['ip'] and target_key['protocol'] == key['protocol']:
                value['direction'] = 'forward'
            else:
                value['direction'] = 'backward'
            self.values[table_num][index].add(value)
            return True

        generate_key = [key, 'stage_1']
        if value == {}:
            value['direction'] = 'forward'    
        success, _ = self._insert_directly(generate_key, value)

        if not success:
            self._rehash()
            success, _ = self._insert_directly(generate_key, value)
        return success
    
    def _insert_directly(self, key: typing.List[typing.Union[dict, str]], value: dict) -> typing.Tuple[bool, list]:
        '''
        function: insert a key into the index table without rehashing
        parameter:
            key: a dictionary containing the keys to be inserted
            value: a ListBuffer containing the values to be inserted
        return:
            is_success: a boolean value indicating whether the insertion is successful
            path: a list containing the path of the key
        '''
        path = []
        current_key = key
        current_value = value
        starting_table = 0
        is_end, is_increase = False, True
        is_Lb = False
        while True:
            for table_id in range(starting_table, 3):
                index = self.hash_functions(current_key[0], table_id)
                if not is_Lb:
                    current_value : dict
                    if self.tables[table_id][index] is None:
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        self.values[table_id][index].add(current_value)
                        is_Lb = True
                        self.num_items += 1
                        is_end = True
                        break
                    else:# random replace
                        # 此处增加随机替换的判定条件 如果满足就替换
                        
                        evicted_key = copy.deepcopy(self.tables[table_id][index])
                        evicted_value = copy.deepcopy(self.values[table_id][index])
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        self.values[table_id][index].add(current_value)
                        current_key = evicted_key
                        path.append(current_key)
                        if evicted_key[1] == 'stage_3':
                            current_value = evicted_value
                            is_end = True
                            is_increase = False
                            break
                        else:
                            evicted_key[1] = stage_name[stage_name.index(evicted_key[1]) + 1]
                            current_value = evicted_value
                            is_Lb = True
                else:
                    current_value : ListBuffer
                    if current_key in path:
                        is_end = True
                        break
                    # 替换并置换现有键
                    if self.tables[table_id][index] is None:
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        for element in current_value.buffer:
                            self.values[table_id][index].add(element)
                        is_Lb = True
                        self.num_items += 1
                        is_end = True
                        break
                    else:
                        evicted_key = copy.deepcopy(self.tables[table_id][index])
                        evicted_value = copy.deepcopy(self.values[table_id][index])
                        self.tables[table_id][index] = current_key
                        self.values[table_id][index] = ListBuffer(self.buffersize, tcp_state=True) if self.type == 'TCP' else ListBuffer(self.buffersize)
                        for element in evicted_value.buffer:
                            self.values[table_id][index].add(element)
                        current_key = evicted_key
                        path.append(current_key)
                        if evicted_key[1] == 'stage_3':
                            current_value = evicted_value
                            is_end = True
                            is_increase = False
                            break
                        else:
                            evicted_key[1] = stage_name[stage_name.index(evicted_key[1]) + 1]
                            current_value = evicted_value 
            if is_end:
                break
        return key not in path, path
    def lookup(self, key: dict) -> typing.Tuple[int, ListBuffer]:
        '''
        parameter:
        
            key: a dictionary containing the keys to be looked up
        return:
            table_num: the table number where the key is found
            index: the index where the key is found
        '''
        for table_id in range(3):
            index = self.hash_functions(key, table_id)
            if self.tables[table_id][index] is not None and match_keys(self.tables[table_id][index][0], key):
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
            if self.tables[table_id][index]is not None and match_keys(self.tables[table_id][index][0], key):
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
    
     