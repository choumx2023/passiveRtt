import ipaddress
import os
import sys
import copy
import typing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import match_keys, protocol_to_int

stage_name = ['stage_1', 'stage_2', 'stage_3']

class ListBuffer:
    '''
    ListBuffer: a class that implements a list-based buffer with a fixed size
    
    '''
    def __init__(self, size):
        
        self.size = size
        self.buffer = []
    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)  # 移除最旧的元素以保持缓冲区大小

    def process_element(self, new_element: list, condition1, condition2, is_add : bool) -> list:
        '''
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
            print('add new element', new_element)
        if first_matched_value is not None and new_element['timestamp'] - first_matched_value['timestamp'] > 1 - 1e-3:
            return None
        
        return first_matched_value
    # typical back-to-back TCP packets
    # 1. C->S: seq = x, ack = y, time = 0.0001
    # 2. C->S: seq = x+50, ack = y, time = 0.0002
    # 3. S->C: seq = y, ack = x+50
    def process_normal_tcp_element(self, new_element : list, is_add : bool) -> list:
        '''
        parameters:
            new_element: value
            is_add: a boolean value indicating whether to add the new element to the buffer
        return:
            first_matched_value: the first matched value in the buffer
        '''
        first_match_value = None
        new_ack = new_element['ack']
        i = len(self.buffer) - 1
        count = 0
        while i >= 0:
            current_element = self.buffer[i]
            if current_element['ack'] < new_element['seq']:
                break
            elif current_element['seq'] == new_ack:
                if first_match_value is None:
                    first_match_value = copy.deepcopy(current_element)
                    count += 1
                else:
                    # 如果不止发送了一个packet，要保留两个
                    if abs(first_match_value['timestamp'] - new_element['timestamp']) < 1e-3 and count == 1:
                        count += 1
                    # 如果连续发送不少于三个，删掉前n-2个
                    else:
                        self.buffer.pop(i)
                        i -= 1
            i -= 1
        if is_add:
            self.add(copy.deepcopy(new_element))
        if first_match_value is not None and new_element['timestamp'] - first_match_value['timestamp'] > 1 - 1e-3:
            return None
        return first_match_value
                        
            
    def print_lb(self):
        for item in self.buffer:
            print(self.buffer)
    def __str__(self) -> str:
        return f"ListBuffer(size={self.size}, buffer={self.buffer})"
    def __repr__(self) -> str:
        return self.__str__()
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
    def __init__(self, initial_size = 100013, buffersize = 1000) -> None:
        '''
        parameter:
            initial_size: the initial size of the hash table
            buffersize: the size of the buffer
        
        '''
        self.size = initial_size
        self.buffersize = buffersize
        self.tables = [[None] * self.size for _ in range(3)]# [None] or [key, stage_name]
        self.values = [[ListBuffer(buffersize) for _ in range(self.size)] for _ in range(3)]
        self.num_items = 0
        self.rehash_threshold = 0.6
        self.max_rehash_attempts = 5
        
    def hash_ip(self, ip : ipaddress.IPv4Address | ipaddress.IPv6Address) -> int:
        ip_int = int(ipaddress.ip_address(ip))
        if ipaddress.ip_address(ip).version == 6:
            ip_int = ip_int % 2**64  # 简化 IPv6 地址
        return ip_int
    def hash_functions(self, key_dict, function_id) -> int:
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
        # 重新哈希的实现
        old_size = self.size
        self.size *= 2
        old_tables = self.tables
        old_values = self.values
        self.tables = [[False, 'no_stage'] * self.size for _ in range(3)]
        self.values = [[ListBuffer(self.buffersize) for _ in range(self.size)] for _ in range(3)]
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
    def insert(self, key: dict, value : ListBuffer = None) -> bool:
        '''
        function: insert a key into the index table , if the insertion fails after multiple attempts and rehashing, return False
        parameter:
            key: a dictionary containing the keys to be inserted
            value: a ListBuffer containing the values to be inserted
        return:
            success: a boolean value indicating whether the insertion is successful
        '''
        if value is None:
            value = ListBuffer(self.buffersize)
        generate_key = [key, 'stage_1']
        success, _ = self._insert_directly(generate_key, value)
        if not success:
            self._rehash()
            success, _ = self._insert_directly(key, value)
        return success
    
    def _insert_directly(self, key: typing.List[typing.Union[dict, str]], value: ListBuffer) -> typing.Tuple[bool, list]:
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
        while True:
            for table_id in range(starting_table, 3):
                index = self.hash_functions(current_key, table_id)
                if self.tables[table_id][index] is None:
                    self.tables[table_id][index] = current_key
                    self.values[table_id][index].add(current_value)
                    self.num_items += 1
                    is_end = True
                    break
                else:
                    # 记录置换路径，用于调试
                    if current_key in path:
                        is_end = True
                        break
                    # 替换并置换现有键
                    evicted_key = copy.deepcopy(self.tables[table_id][index])
                    evicted_value = copy.deepcopy(self.values[table_id][index])
                    self.tables[table_id][index] = current_key
                    self.values[table_id][index].add(current_value)
                    current_key = evicted_key
                    path.append(current_key)
                    if evicted_key[1] == 'stage_3':
                        current_value = evicted_value
                        starting_table = 0
                        is_end = True
                        is_increase = False
                        break
                    else:
                        evicted_key[1] = stage_name[stage_name.index(evicted_key[1]) + 1]
                        current_value = evicted_value
            if is_end:
                break
        return is_increase, path
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
            if self.tables[table_id][index] is not None and self.tables[table_id][index][0] == key:
                return table_id, self.values[table_id][index]
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
            if self.tables[table_id][index][0] == True and self.tables[table_id][index] == key:
                
                # Remove the key and value
                # self.values[table_id][index].clear()
                self.tables[table_id][index] = [False, 'no_stage']
                self.num_items -= 1
                return True
        return False  # Key not found

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
    
    
     