import time

from utils import utils
import ipaddress
from scapy.all import IP, IPv6
class RTTTable:
    '''
    A class to store and calculate RTT samples for IP address pairs.
    key : (src_ip, dst_ip)
    value : list of RTT samples, each containing 'rtt', 'timestamp' and 'type'
    '''
    def __init__(self):
        # 使用字典存储RTT样本，键为IP地址对，值为包含RTT样本的列表
        self.rtt_samples = {}
        self.max_time = None
        self.min_time = None
    def _get_ip_pair_key(self, src_ip, dst_ip) -> str:
        """生成IP对的键，支持IPv4和IPv6"""
        try:
            src_ip = ipaddress.ip_address(src_ip)
            dst_ip = ipaddress.ip_address(dst_ip)
        except ValueError:
            raise ValueError("Invalid IP addresses.")

        # 将IP地址转换为统一的二进制形式比较
        if src_ip > dst_ip:
            return f"{src_ip}-{dst_ip}"
        else:
            return f"{dst_ip}-{src_ip}"

    # 接受源IP地址、目标IP地址、RTT值和时间戳作为参数，并将新的RTT样本添加到RTT表中
    def add_rtt_sample(self, src_ip, dst_ip, rtt, timestamp,types = None,  direction ='forward', extra_data = None ) -> None:
        """向RTT表中添加一个新的RTT样本"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples:
            self.rtt_samples[key] = []
        self.rtt_samples[key].append({'rtt': rtt, 'timestamp': timestamp, 'types': types, 'direction': direction, 'extra_data': extra_data})

    def calculate_average_rtt(self, src_ip, dst_ip) -> float:
        """计算指定IP对的平均RTT"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key in self.rtt_samples and len(self.rtt_samples[key]) > 0:
            return sum(self.rtt_samples[key]) / len(self.rtt_samples[key])
        return None
    def calculate_continue_rtt(self, src_ip, dst_ip) -> list:
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key in self.rtt_samples and len(self.rtt_samples[key]) > 0:
            return self.rtt_samples[key]
        return None
    def calculate_and_add_rtt(self, src_ip, dst_ip, request_timestamp : float, response_timestamp : float) -> None:
        """根据请求和响应的时间戳计算RTT，并添加到表中"""
        rtt = response_timestamp - request_timestamp
        self.add_rtt_sample(src_ip, dst_ip, rtt, response_timestamp)

    def get_average_rtt(self, src_ip, dst_ip) -> float:
        """计算并返回指定IP地址对的平均RTT"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples or len(self.rtt_samples[key]) == 0:
            return None

        total_rtt = sum(sample['rtt'] for sample in self.rtt_samples[key])
        return total_rtt / len(self.rtt_samples[key])

    def clean_old_samples(self, threshold_seconds):
        """清理旧的RTT样本，只保留指定时间阈值内的样本"""
        current_time = time.time()
        for key in list(self.rtt_samples.keys()):
            self.rtt_samples[key] = [sample for sample in self.rtt_samples[key] if current_time - sample['timestamp'] <= threshold_seconds]
            if not self.rtt_samples[key]:
                del self.rtt_samples[key]
    def print_rtt(self) -> None:
        for key in self.rtt_samples:
            print(key)
            for sample in self.rtt_samples[key]:
                print(f"RTT: {sample['rtt']} seconds, Timestamp: {sample['timestamp']}, types: {sample['types']}, direction: {sample['direction']}, extra_data: {sample['extra_data']}")
    def print_tcprtt(self) -> None:
        for key in self.rtt_samples:
            print(key)
            for sample in self.rtt_samples[key]:
                print(f"RTT: {sample['rtt']} ms, Timestamp: {sample['timestamp']}, types: {sample['types']}, direction: {sample['direction']}")
    def print_tcptrace(self):
        for key in self.rtt_samples:
            print(key, self.rtt_samples[key])  
    def add_tcptrace_samples(self, src_ip, dst_ip, value: dict, types = None) -> None:
        """向RTT表中添加一个新的RTT样本"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples:
            self.rtt_samples[key] = []
        self.rtt_samples[key].append(value)

    '''rtt_table = RTTTable()

    # 示例数据包时间戳
    rtt_table.calculate_and_add_rtt('192.168.1.1', '8.8.8.8', 1625140800, 1625140810)
    rtt_table.calculate_and_add_rtt('192.168.1.1', '8.8.8.8', 1625140820, 1625140830)

    # 获取平均RTT
    average_rtt = rtt_table.get_average_rtt('192.168.1.1', '8.8.8.8')
    print(f"Average RTT for 192.168.1.1 to 8.8.8.8: {average_rtt} seconds")

    # 清理旧样本（示例中阈值设为30天）
    rtt_table.clean_old_samples(30 * 24 * 60 * 60)
    '''
    def calc_rtt_cdf(self):
        rtt_list = []
        for key in self.rtt_samples:
            for sample in self.rtt_samples[key]:
                rtt_list.append(sample['rtt'])
        rtt_list.sort()
        return rtt_list
        
 