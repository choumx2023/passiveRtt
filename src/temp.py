class CompressedIPNode:
    def __init__(self, network):
        self.network = network
        self.children = {}
        self.parent = None  # 指向父节点
        self.stats = defaultdict(int)  # 统计信息

    def record_activity(self, activity_type, count=1):
        # 在当前节点记录活动
        self.stats[activity_type] += count
        # 递归更新父节点的统计信息
        if self.parent:
            self.parent.record_activity(activity_type, count)
def record_activity(self, ip, activity_type, count=1, timestamp=None):
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node:
            node.record_activity(activity_type, count, timestamp)
            
def record_activity(self, activity_type, count=1, timestamp=None):
        # 添加计数和时间戳
        self.stats[activity_type]['count'] += count
        if timestamp:
            self.stats[activity_type]['timestamps'].append(timestamp)
        # 递归更新父节点
        if self.parent:
            self.parent.record_activity(activity_type, count, timestamp)
def query_activity(self, activity_type):
        # 获取活动的计数和所有相关时间戳
        count = self.stats[activity_type]['count']
        timestamps = self.stats[activity_type]['timestamps']
        return count, timestamps