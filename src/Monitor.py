import sys
sys.path.append('/Users/choumingxi/Documents/GitHub/newrtt/src')
import ipaddress
from collections import defaultdict, Counter
import time
import statistics
from random import randint
import logging
import os
import pickle
import math
import time
import seaborn
import math
from collections import deque
class WelfordVariance:
    '''
    This class implements the Welford algorithm for calculating the variance of a stream of data points.
    
    '''
    def __init__(self, time_window : int = 1200, max_count : int = 120):
        '''
        params:
            time_window: 一个时间窗口，用于限制数据点的时间范围。
            max_count: 一个最大计数，用于限制数据点的数量。
        设置时间窗口和最大计数，以便在超出时间窗口或计数时移除数据点。
        '''
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.time_window = time_window
        self.max_count = max_count
        self.data = deque()

    def update(self, x : float, timestamp : float):
        '''
        params:
            x: 新的数据点 
            timestamp: 时间戳
        更新统计数据，包括计数、均值和方差。
        '''
        self.data.append((x, timestamp))
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def remove(self, x : float):
        '''
        params:
            x: 要移除的数据点
        移除指定的数据点，并更新统计数据。
        '''
        if self.count <= 1:
            self.count = 0
            self.mean = 0
            self.M2 = 0
        else:
            self.count -= 1
            delta = x - self.mean
            self.mean -= delta / self.count
            delta2 = x - self.mean
            self.M2 -= delta * delta2

    def variance(self):
        '''
        params:
            None
        返回方差。
        
        '''
        if self.count < 2:
            return float('nan')
        return self.M2 / (self.count - 1)

    def str_variance(self):
        return f'Count: {self.count}, Variance: {self.variance()}, Mean: {self.mean}'

    def remove_outdated(self, current_time):
        '''
        params:
            current_time: 当前时间戳
        移除超出时间窗口的数据点。
        '''
        # 移除超出时间窗口的数据点
        while self.time_window and self.data and (current_time - self.data[0][1]) > self.time_window:
            old_value, _ = self.data.popleft()
            self.remove(old_value)

    def check_anomalies(self, newrtt, timestamp):
        current_time = timestamp
        # 移除超出时间窗口的数据点
        self.remove_outdated(current_time)
        
        # 检查新数据点是否异常
        if self.count >= 6 and newrtt - self.mean >  3 + 0.95 * math.sqrt(self.variance()):
            return True
        
        # 如果新数据点不是异常值，则更新统计数据
        self.update(newrtt, timestamp)
        return False

# 使用示例
def default_state():
    return {
        'count': 0,
        'timestamps': []
    }
class CompressedIPNode:
    '''
    This class represents a node in a compressed trie for storing IP addresses. It supports both IPv4 and IPv6 addresses.
    It contains methods for recording RTT values and network activity, as well as aggregating statistics and detecting anomalies.
    '''
    def __init__(self, network : ipaddress.IPv4Address|ipaddress.IPv6Address, logger : str=None):
        '''
        params:
            network: IP地址的网络范围
            logger: 日志记录器
        初始化一个新的IP节点。
        '''
        self.network = network # 记录当前节点的网络范围
        self.subnets = []  # 记录由哪些子网合并而来
        self.children = {} # 存储子节点
        self.parent = None  # 记录父节点
        self.alerts = []
        self.logger = logger
        self.contain_ip_number = 0
        self.contain_rtt_ip_number = 0
        self.stats = defaultdict(default_state)
        self.rtt_records = defaultdict(list) # 正常的rtt记录
        self.all_rtt_records = [] # 所有正常的rtt记录
        self.anomalous_rtts_records = defaultdict(list) # 异常的rtt记录
        self.rtt_stats = {
            'min_rtt': float('inf'),
            'max_rtt': float('-inf'), 
        }
        self.rtt_WelfordVariance = WelfordVariance(time_window=1000, max_count=1000)
        self._stats_dirty = True
        self._rtt_dirty = True
    def aggregate_stats(self):
        '''
        聚合来自所有子节点的stats数据。
        '''
        return 
        if not self._stats_dirty:
            return # 如果没有新的统计数据，不需要重新聚合
        """聚合来自所有子节点的统计数据。"""
        aggregated_stats = defaultdict(default_state)
        for child in self.children.values():
            for key, value in child.stats.items():
                aggregated_stats[key]['count'] += value['count']
                aggregated_stats[key]['timestamps'].extend(value['timestamps'])
        self.stats = aggregated_stats
        self._stats_dirty = False
    def is_rtt_anomalous(self,  rtt, timestamp):
        '''
        params:
            rtt: RTT值
            timestamp: 时间戳
        返回RTT是否异常。
        '''
        # TODO: 实现检测RTT是否异常的逻辑
        if self.rtt_WelfordVariance.check_anomalies(rtt, timestamp):
            return True
        return False
    def aggregate_rtt(self):
        '''
        params:
            None
        聚合来自所有子节点的RTT数据。
        '''
        aggregated_rtt = defaultdict(list)
        for child in self.children.values():
            if len(child.rtt_records) < 5:
                continue
            
            for key, values in child.rtt_records.items(): # 只收集有效的rtt记录
                aggregated_rtt[key].extend(values)
        self.rtt_records = aggregated_rtt
    def record_rtt(self, protocol, pattern, rtt, timestamp,  check_anomalies=False):
        '''
        params:
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
            check_anomalies: 是否检查异常
        记录RTT值。
        '''
        if not self._rtt_dirty:
            return
        key = (protocol, pattern)
        if check_anomalies and self.is_rtt_anomalous(rtt, timestamp): # 检查RTT是否异常，需要设置check_anomalies=True
            self.anomalous_rtts_records[key].append((rtt, timestamp))
            if self.logger:
                self.logger.warning(f'Anomalous RTT detected: {protocol} - {rtt}ms at {timestamp}')
        else: # 如果RTT正常，则记录到正常的rtt记录中
            if check_anomalies == False:
                self.rtt_WelfordVariance.update(rtt, timestamp)
            self.logger.info(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
            self.rtt_records[key].append((rtt, timestamp))
            self.all_rtt_records.append((rtt, timestamp))
            if rtt < self.rtt_stats['min_rtt'] and rtt > 0 :
                self.rtt_stats['min_rtt'] = rtt
            if rtt > self.rtt_stats['max_rtt'] and rtt < 1e4:
                self.rtt_stats['max_rtt'] = rtt
                
            if self.logger:
                self.logger.debug(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
            # 父母就不检查了    
            if self.parent and len(self.all_rtt_records) > 5:
                self.parent.upstream_rtt(protocol, pattern, rtt, timestamp)
    def upstream_rtt(self, protocol, pattern, rtt, timestamp):
        '''
        This function records the upstream RTT values, it delivers the RTT values to the parent node.
        params:
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
        上游RTT。
        '''
        key = (protocol, pattern)
        self.logger.info(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
        self.rtt_records[key].append((rtt, timestamp))
        self.all_rtt_records.append((rtt, timestamp))
        if rtt < self.rtt_stats['min_rtt'] and rtt > 0 :
            self.rtt_stats['min_rtt'] = rtt
        if rtt > self.rtt_stats['max_rtt'] and rtt < 1e4:
            self.rtt_stats['max_rtt'] = rtt
        if self.parent:
            self.parent.upstream_rtt(protocol, pattern, rtt, timestamp)
    def record_activity_recursive(self, protocol, action, count=1, timestamp=None, check_anomalies=False):
        '''
        This function records activity recursively.
        params:
            protocol: 协议
            action: 动作
            count: 数量
            timestamp: 时间戳
            check_anomalies: 是否检查异常
        递归记录活动。
        '''
        # Only check anomalies if flag is True
        key = (protocol, action)
        self.stats[key]['count'] += count
        if timestamp:
            self.stats[key]['timestamps'].append(timestamp)
        if check_anomalies:
            self.detect_protocols_anomalie(protocol)

        # Decide whether to check for anomalies only once at the initial call
        if self.parent:
            self.parent.record_activity_recursive(protocol, action, count, timestamp, check_anomalies)
    def calculate_contain_ip_number(self):
        '''
        This function calculates the number of IP addresses contained in the network.
        params:
            None
        returns:
            contain_ip_number(int): The number of IP addresses contained in the network.
        '''
        return self.contain_ip_number
    def update_contain_ip_number(self):
        if self.network.prefixlen == 32 and self.network.version == 4:
            self.contain_ip_number = 1
            if len(self.rtt_records) > 5:
                self.contain_rtt_ip_number = 1
        elif self.network.prefixlen == 128 and self.network.version == 6:
            self.contain_ip_number = 1
            if len(self.rtt_records) > 5:
                self.contain_rtt_ip_number = 1
        else:
            self.contain_ip_number = sum([child.calculate_contain_ip_number() for child in self.children.values()])
            self.contain_rtt_ip_number = sum([child.contain_rtt_ip_number for child in self.children.values()])
    def detect_protocols_anomalie(self, protocol):
        '''
        This function detects anomalies in the network traffic.
        havent been implemented yet.
        
        
        params:
            protocol: The protocol to check for anomalies
        returns:
            None
        '''
        # 示例：检测DNS请求和响应的数量差异
        if protocol == "DNS":
            requests = self.stats[("DNS", "Query")]['count']
            responses = self.stats[("DNS", "Response")]['count']
            if abs(requests - responses) > 100:  # 假设差异阈值为100
                anomaly = f"Anomaly detected: DNS requests ({requests}) and responses ({responses}) difference exceeds threshold"
                self.alerts.append(anomaly)
                print(anomaly)  # 或者将异常信息发送到日志系统或警报系统
        
    def get_rtt_stats(self):
        '''
        This function returns the RTT statistics for the network.
        params:
            None
        returns:
            rtt_stats(dict): The RTT statistics for the network
        '''
        all_rtts = []
        for rtts in self.rtt_records.values():
            all_rtts.extend([rtt for rtt, _ in rtts])
        if not all_rtts:
            return None
        return {
            'min_rtt': self.rtt_stats['min_rtt'],
            'max_rtt': self.rtt_stats['max_rtt'],
            'all_rtts': all_rtts
        }

    def get_subnets_rtt_stats(self):
        '''
        This function returns the RTT statistics for the subnets.
        params:
            None
            
        have not been implemented yet.
        '''
        min_rtt, max_rtt, all_rtts = float('inf'), float('-inf'), []
        for child in self.children.values():
            stats = child.get_rtt_stats()
            if stats:
                min_rtt = min(min_rtt, stats['min_rtt'])
                max_rtt = max(max_rtt, stats['max_rtt'])
                all_rtts.extend(stats['all_rtts'])
        if min_rtt == float('inf') or max_rtt == float('-inf'):
            return None
        return {
            'min_rtt': min_rtt,
            'max_rtt': max_rtt,
            'all_rtts': all_rtts
        }
    def print_node(self):
        print('my network:', self.network, 'my stats:', self.stats)

    def __rtt__(self, prefix=''):
        prefix1 = prefix + '  '
        if self.rtt_records == {}:
            return f'{prefix}RTT Datas : \n{prefix1}No RTT data'
        def format_output(rtts):
            #return f'\n{prefix1}  '.join([f'{rtt}ms, {timestamps}' for rtt, timestamps in rtts])
            return f'\n{prefix1}  '.join([', '.join(map(str, rtts[i:i+8])) for i in range(0, len(rtts), 8)])
        rtt_info = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT : \n  {prefix1}{format_output(rtts)}'
            for (protocol, pattern), rtts in self.rtt_records.items())
        return f'{prefix}RTT Datas :\n {rtt_info}'
    def __anormalies__(self, prefix=''):
        '''
        params:
            prefix: 前缀
        返回异常数据。
        '''
        prefix1 = prefix + '  '
        if self.anomalous_rtts_records == {}:
            return f'{prefix}Anomalies Data: \n{prefix1}No anomalies data'
        def format_output(rtts):
            #return f'\n{prefix1}  '.join([f'{rtt}ms, {timestamps}' for rtt, timestamps in rtts])
            return f'\n{prefix1}  '.join([', '.join(map(str, rtts[i:i+8])) for i in range(0, len(rtts), 8)])
        rtt_info = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT : \n  {prefix1}{format_output(rtts)}'
            for (protocol, pattern), rtts in self.anomalous_rtts_records.items())
        return f'{prefix}Anomalies Data: \n {rtt_info}'
    def __stats__(self, prefix=''):
        prefix1 = prefix + '  '
        if self.stats == {}:
            return f'{prefix}Stats Data: \n{prefix1}No stats data'
        def format_timestamps(timestamps):
            # 将时间戳分段，每段最多包含5个时间戳
            return f'\n{prefix1}  '.join([', '.join(map(str, timestamps[i:i+10])) for i in range(0, len(timestamps), 10)])
        stats_info = '\n'.join(
            f'{prefix1}{protocol}={action} : {details["count"]}\n {prefix1} {format_timestamps(details["timestamps"])}'
            for (protocol, action), details in self.stats.items())
        return f'{prefix}Stats Data: \n{stats_info}'

    def __basic__(self, prefix=''):
        if prefix:
            prefix = '|' + '-' * (len(prefix) - 1)
        wol = self.rtt_WelfordVariance.str_variance()
        basic_info = f"{prefix}{self.network}, : IP Count={self.contain_ip_number, self.contain_rtt_ip_number}, RTT range: {self.rtt_stats['min_rtt']}ms - {self.rtt_stats['max_rtt']}ms, wolvalue = {wol},logger = {self.logger}"
        return basic_info
    
class CompressedIPTrie:
    '''
    This class represents a compressed trie for storing IP addresses. It supports both IPv4 and IPv6 addresses.
    '''
    def __init__(self, ip_version=4, logger=None):
        '''
        This function initializes a new compressed trie for storing IP addresses.
        params:
            ip_version: The IP version (4 or 6)
            logger: The logger object
        return:
            None
        '''
        self.root = CompressedIPNode(network=ipaddress.ip_network("0.0.0.0/0" if ip_version == 4 else "::/0", ), logger=logger)
        self.logger = logger    
        self.ip_version = ip_version
    def add_ip(self, ip : str):
        '''
        This function adds an IP address to the trie. If the IP address is already in the trie, it will not be added again, otherwise it will be created and added.
        If it is a new node, then it will be merged with its parent node if possible.
        
        params:
            ip: The IP address to add
        returns:
            None
        '''
    # 确保IP版本匹配
        if ipaddress.ip_address(ip).version != self.ip_version:
            return
        new_ip = f'{ip}/32' if self.ip_version == 4 else f'{ip}/128'
        new_net = ipaddress.ip_network(new_ip, strict=False)
        if not self.find_node(ip):
            new_node = CompressedIPNode(network=new_net, logger=self.logger)
            new_node.update_contain_ip_number()
            self.insert_network(new_node)
            self._merge_network(new_node)

    def insert_network(self, new_net: CompressedIPNode):
        '''
        This function inserts a new network into the trie. It will recursively find the correct insertion point.
        params:
            new_net: The new IP node
        returns:
            None     
        '''
        node = self.root
        # 需要递归查找正确的插入点
        while True:
            placed = False
            for child in list(node.children.values()):
                if new_net.network.subnet_of(child.network):
                    node = child
                    placed = True
                    break
            if not placed:
                break
        new_net.parent = node
        node.children[new_net.network] = new_net
        node.update_contain_ip_number()
    
    def _merge_network(self, new_net : CompressedIPNode) -> None:
        '''
        This function merges networks if possible. It will recursively merge networks up to the root node.
        params:
            new_net: The new IP node
        returns:
            None
        
        '''
        # 合并网络应当考虑可能需要递归上溯到不只是直接父节点
        step = 2 if self.ip_version == 4 else 4
        merge_count = 2 if self.ip_version == 4 else 4
        max_subnet =  8 if self.ip_version == 4 else 32
        parent = new_net.parent
        while parent and new_net.network.prefixlen <= max_subnet:
            super_net = new_net.network.supernet(new_prefix=new_net.network.prefixlen - step)
            if super_net.prefixlen % step != 0:
                print('Invalid supernet.')
                break
            elif super_net == parent.network:
                break
            # 检查是否有足够的子节点可以合并
            max_subnet_prefix = super_net.prefixlen - step
            eligible_children = [child for child in parent.children.values() if child.network.subnet_of(super_net)]
            max_subnet_children = [child for child in parent.children.values() if child.network.subnet_of(super_net) and child.network.prefixlen == max_subnet_children]
            # TODO : 可以根据需要调整合并的条件 首先就是根据子网的rtt数量 这里需要一个聚类方法，参考指标就是各个子网的rtt_stats
            
            
            if len(max_subnet_children) >= merge_count:
                supernet_node = CompressedIPNode(network=super_net, logger=self.logger)
                parent.children[super_net] = supernet_node
                for child in eligible_children:
                    del parent.children[child.network]
                    supernet_node.children[child.network] = child
                    child.parent = supernet_node
                    supernet_node.subnets.append(child.network)
                supernet_node.aggregate_stats()
                supernet_node.aggregate_rtt()   
                supernet_node.update_contain_ip_number()
                supernet_node.parent = parent
                parent = supernet_node.parent
                new_net = supernet_node
            else:
                break

    def find_node(self, ip: str) -> CompressedIPNode:
        '''
        This function finds the node in the trie that contains the given IP address.
        params:
            ip: The IP address to find
        returns:
            node(CompressedIPNode): The node that contains the IP address
        '''
        node = self.root
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version != self.ip_version:
            return None
        target = ipaddress.ip_network(f'{ip}/{32 if ip_obj.version == 4 else 128}', strict=False)

        while True:
            found = False
            for child in node.children.values():
                if child.network == target:
                    return child
                elif target.subnet_of(child.network):
                    node = child
                    found = True
                    break
            if not found:
                return None
    def record_activity(self, ip : str, activity_type : str, count=1, timestamp=None):
        '''
        This function records activity for a given IP address. It will find the node that contains the IP address and record the activity.
        If the IP address is not in the trie, it will be added first.
        params: 
            ip: The IP address
            activity_type: The type of activity
            count: The count of the activity
            timestamp: The timestamp of the activity
        returns:
            None
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        trie : CompressedIPTrie
        node = trie.find_node(ip)
        if node == None:
            trie.add_ip(ip)
            node = trie.find_node(ip)
            
        if node:
            node.record_activity(activity_type, count, timestamp)
    def print_tree(self, node=None, indent=0, file_path='tree.txt'):
        '''
        params:
            node: 节点
            indent: 缩进
            file_path: 文件路径
        打印树。
        '''
        if node is None:
            node = self.root

        indent_str = '    ' * indent
        with open(file_path, 'a') as f:
            f.write(f'{node.__basic__(indent_str)}\n')
            f.write(f"{node.__stats__(indent_str)}\n")
            f.write(f"{node.__rtt__(indent_str)}\n")
            f.write(f'{node.__anormalies__(indent_str)}\n')
        for child in node.children.values():
            self.print_tree(child, indent + 1, file_path)
    def collect_smallest_subnets(self):
        smallest_subnets = []
        self._collect_smallest_subnets_helper(self.root, smallest_subnets)
        return smallest_subnets

    def _collect_smallest_subnets_helper(self, node, smallest_subnets):
        # Define what is considered a "smallest subnet"
        if (self.ip_version == 4 and node.network.prefixlen == 32) or (self.ip_version == 6 and node.network.prefixlen == 128):
            smallest_subnets.append(node)
        for child in node.children.values():
            self._collect_smallest_subnets_helper(child, smallest_subnets)
class NetworkTrafficMonitor:
    def __init__(self, name = '', check_anomalies = 'True', logger = None):
        '''
        params:
            name: 名称
            check_anomalies: 是否检查异常
            logger: 日志记录器
        初始化网络流量监控器。
        '''
        self.ipv4_trie = CompressedIPTrie(ip_version=4, logger=logger)
        self.ipv6_trie = CompressedIPTrie(ip_version=6, logger=logger)
        self.timeslot = 0.2
        self.suffix = name
        self.check_anomalies = check_anomalies
    def add_ip_and_record_activity(self, ip, protocol, action, count=1, timestamp=None):
        '''
        params:
            ip: IP地址
            protocol: 协议
            action: 动作
            count: 数量
            timestamp: 时间戳
        添加IP地址并记录活动。
        '''
        # 确定使用IPv4还是IPv6的Trie
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        
        # 尝试找到已存在的节点
        node = trie.find_node(ip)
        
        # 如果节点不存在，先添加IP
        if not node:
            trie.add_ip(ip)
            node = trie.find_node(ip)  # 重新获取新添加的节点
        
        # 记录活动，更新节点及其所有父节点
        if node:
            node.record_activity_recursive(protocol, action, count, timestamp)
    def query_rtt(self, ip, protocol):
        '''
        params:
            ip: IP地址
            protocol: 协议
        查询RTT。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node and protocol in node.rtt_records:
            return statistics.mean(node.rtt_records[protocol]) if node.rtt_records[protocol] else None
        return None
    def query_activity(self, ip, protocol, action):
        '''
        params:
            ip: IP地址
            protocol: 协议
            action: 动作
        查询活动。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node:
            key = (protocol, action)
            total_count = sum(item[0] for item in node.stats[key])  # 汇总所有count
            timestamps = [item[1] for item in node.stats[key]]  # 获取所有时间戳
            return total_count, timestamps
        return 0, []
    def add_or_update_ip_with_rtt(self, ip, protocol, pattern, rtt, timestamp):
        '''
        params:
            ip: IP地址
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
        添加或更新IP地址和RTT。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        # 如果IP节点不存在，则添加它
        if not node:
            trie.add_ip(ip)
            node = trie.find_node(ip)  # 确保节点被添加后重新获取它
        # 现在记录RTT数据，假设节点现在肯定存在
        if node:
            node.record_rtt(protocol, pattern, rtt, timestamp, check_anomalies= self.check_anomalies)
        # 可选：检测异常情况
        if 0:
            node.check_rtt_anomalies()
    def detect_attack(self, ip, threshold=1000):
        '''
        params:
            ip: IP地址
            threshold: 阈值
        检测攻击。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node:
            total_requests = sum(node.stats.values())
            return total_requests > threshold
        return False
    def print_trees(self):
        with open(f'{self.suffix}_tree.txt', 'w') as f:
            f.write("")  # Clear the contents of the file
        self.ipv4_trie.print_tree(file_path=f'{self.suffix}_tree.txt')
        #print("IPv4 Trie is saved to tree.txt")
        self.ipv6_trie.print_tree(file_path=f'{self.suffix}_tree.txt')
        #print("IPv6 Trie is saved to tree.txt")
    def save_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    def merge_nodes(self, node_a, node_b):
        # 合并统计数据和RTT记录
        for key, stats in node_b.stats.items():
            if key in node_a.stats:
                node_a.stats[key]['count'] += stats['count']
                node_a.stats[key]['timestamps'].extend(stats['timestamps'])
            else:
                node_a.stats[key] = stats

        for key, rtts in node_b.rtt_records.items():
            if key in node_a.rtt_records:
                node_a.rtt_records[key].extend(rtts)
            else:
                node_a.rtt_records[key] = rtts

        # 递归合并子节点
        for subnet, child_node_b in node_b.children.items():
            if subnet in node_a.children:
                self.merge_nodes(node_a.children[subnet], child_node_b)
            else:
                node_a.children[subnet] = child_node_b
                child_node_b.parent = node_a

    def merge_smallest_network(self, node_parent, ip_version=4):
        prefix = 32 if ip_version == 4 else 128
        trie = self.ipv4_trie if ip_version == 4 else self.ipv6_trie

        for child in node_parent.children.values():
            if child.network.prefixlen == prefix:
                target_node = trie.find_node(str(child.network.network_address))
                if target_node is None:
                    trie.add_ip(str(child.network.network_address))
                    target_node = trie.find_node(str(child.network.network_address))
                self.merge_nodes(target_node, child)
            else:
                self.merge_smallest_network(child, ip_version)

    def merge_monitor(self, other_monitor):
        # 合并 IPv4 和 IPv6 Trie 的根节点
        self.merge_smallest_network(other_monitor.ipv4_trie.root, ip_version=4)
        self.merge_smallest_network(other_monitor.ipv6_trie.root, ip_version=6)
    
    @staticmethod
    def load_state(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    def clear_state(self):
        self.ipv4_trie = CompressedIPTrie(ip_version=4, logger=self.logger)
        self.ipv6_trie = CompressedIPTrie(ip_version=6, logger=self.logger)
    def __repr__(self) -> str:
        return (f'NetworkTrafficMonitor({self.ipv4_trie}, {self.ipv6_trie})')
    def __str__(self) -> str:
        return (f'NetworkTrafficMonitor({self.ipv4_trie}, {self.ipv6_trie})')

def generate_single_ip(base_net, prefix):
    '''
    params:
        base_net: 基础网络
        prefix: 前缀
    生成一个随机IP地址，基于指定的基础网络和子网掩码。
    '''
    """生成一个随机IP地址，基于指定的基础网络和子网掩码。"""
    network = ipaddress.ip_network(base_net)
    subnets = list(network.subnets(new_prefix=prefix))
    subnet = subnets[randint(0, len(subnets) - 1)]
    ip = str(subnet[randint(1, subnet.num_addresses - 2)])
    return ip
def setup_logging():
    '''
    params:
        None
    设置日志记录器。
    '''
    log_directory = "./logs"
    log_path = os.path.join(log_directory, f"network_monitor.log")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger()

# 调用 setup_logging 函数以初始化日志设置
def read_pickle(filename):
    '''
    params:
        filename: 文件名
    读取pickle文件。
    '''
    with open (filename, 'rb') as f:
        data = pickle.load(f)
        return data
def main():
    logger = setup_logging()
    monitor = NetworkTrafficMonitor(logger = logger)
    
    monitor.add_ip_and_record_activity('192.168.1.200', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.100', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.201', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.101', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.202', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.102', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.203', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.103', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.204', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.104', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.205', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.105', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_or_update_ip_with_rtt('192.168.1.105', 'TCP', 'SYN',  700, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.206', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.106', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.207', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.108', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.210', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.109', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('192.168.1.230', 'DNS', 'Query', 1, time.time())
    monitor.add_ip_and_record_activity('192.168.1.230', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.110', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('192.168.1.120', 'TCP', 'forward',1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.120', 'TCP', 'Timestamp',  300, time.time())



    monitor.add_ip_and_record_activity('2001:db8::1', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::2', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::3', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::4', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::5', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::6', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::7', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::8', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::9', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::10', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::10', 'TCP', 'SYN',  700, time.time())
    monitor.add_ip_and_record_activity('2001:db8::11', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::12', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::13', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::14', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::15', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::16', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::17', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::18', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::19', 'DNS', 'Query', 1, time.time())
    monitor.add_ip_and_record_activity('2001:db80:0000::20', 'TCP', 'forward', 2, time.time())
    # monitor.add_record('DNS', 'Query', num = 10, timestamps)
    # 打印Trie树以查看子网聚合情况
    monitor.print_trees()
    monitor.save_state('network_monitor.pkl')

def test():
    monitor = read_pickle('current_monitor.pkl')
    print(monitor)
    monitor.print_trees()
if __name__ == "__main__":
    test()


