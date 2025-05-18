import time
from typing import Optional, List, Dict, Union, Any

from utils import utils
import ipaddress
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6

class RTTTable:
    '''
    A class to store and calculate RTT samples for IP address pairs.
    key : (src_ip, dst_ip)
    value : list of RTT samples, each containing 'rtt', 'timestamp' and other metadata
    '''

    def __init__(self) -> None:
        # 使用字典存储RTT样本，键为IP地址对，值为包含RTT样本的列表
        self.rtt_samples: dict[str, list[dict[str, Any]]] = {}
        self.max_time: Optional[float] = None
        self.min_time: Optional[float] = None

    def _get_ip_pair_key(self, src_ip: str, dst_ip: str) -> str:
        """生成IP对的键，支持IPv4和IPv6"""
        try:
            src_ip_obj = ipaddress.ip_address(src_ip)
            dst_ip_obj = ipaddress.ip_address(dst_ip)
        except ValueError:
            raise ValueError("Invalid IP addresses.")

        # 将IP地址转换为统一的二进制形式比较，保证键的一致性
        if int(src_ip_obj) > int(dst_ip_obj):
            return f"{src_ip}-{dst_ip}"
        else:
            return f"{dst_ip}-{src_ip}"

    # 添加相关方法
    def add_rtt_sample(self, src_ip: str, dst_ip: str, rtt: float, timestamp: float,
                       types: Optional[str] = None, direction: str = 'forward', extra_data: Optional[dict] = None) -> None:
        """向RTT表中添加一个新的RTT样本"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples:
            self.rtt_samples[key] = []
        self.rtt_samples[key].append({
            'rtt': rtt,
            'timestamp': timestamp,
            'types': types,
            'direction': direction,
            'extra_data': extra_data
        })

    def add_tcptrace_samples(self, src_ip: str, dst_ip: str, value: dict, types: Optional[str] = None) -> None:
        """向RTT表中添加一个新的TCP trace样本，保持兼容性"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples:
            self.rtt_samples[key] = []
        self.rtt_samples[key].append(value)

    # 计算相关方法
    def calculate_and_add_rtt(self, src_ip: str, dst_ip: str, request_timestamp: float, response_timestamp: float) -> None:
        """根据请求和响应的时间戳计算RTT，并添加到表中"""
        rtt = response_timestamp - request_timestamp
        self.add_rtt_sample(src_ip, dst_ip, rtt, response_timestamp)

    def get_average_rtt(self, src_ip: str, dst_ip: str) -> float | None:
        """计算并返回指定IP地址对的平均RTT"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key not in self.rtt_samples or len(self.rtt_samples[key]) == 0:
            return None
        total_rtt = sum(sample['rtt'] for sample in self.rtt_samples[key])
        return total_rtt / len(self.rtt_samples[key])

    def get_all_samples(self, src_ip: str, dst_ip: str) -> list[dict[str, Any]] | None:
        """获取指定IP对的所有RTT样本"""
        key = self._get_ip_pair_key(src_ip, dst_ip)
        if key in self.rtt_samples and len(self.rtt_samples[key]) > 0:
            return self.rtt_samples[key]
        return None

    # 清理相关方法
    def clean_old_samples(self, threshold_seconds: float) -> None:
        """清理旧的RTT样本，只保留指定时间阈值内的样本"""
        current_time = time.time()
        for key in list(self.rtt_samples.keys()):
            self.rtt_samples[key] = [
                sample for sample in self.rtt_samples[key]
                if current_time - sample['timestamp'] <= threshold_seconds
            ]
            if not self.rtt_samples[key]:
                del self.rtt_samples[key]

    # 打印相关方法
    def print_rtt(self) -> None:
        """打印所有RTT样本，单位为秒，包含详细信息"""
        for key in self.rtt_samples:
            print(f"IP Pair: {key}")
            for sample in self.rtt_samples[key]:
                print(f"  RTT: {sample['rtt']} seconds, Timestamp: {sample['timestamp']}, "
                      f"Types: {sample['types']}, Direction: {sample['direction']}, Extra Data: {sample['extra_data']}")

    def print_tcprtt(self) -> None:
        """打印所有TCP RTT样本，单位为毫秒，包含类型和方向"""
        for key in self.rtt_samples:
            print(f"IP Pair: {key}")
            for sample in self.rtt_samples[key]:
                print(f"  RTT: {sample['rtt']} ms, Timestamp: {sample['timestamp']}, "
                      f"Types: {sample['types']}, Direction: {sample['direction']}")

    def print_tcptrace(self) -> None:
        """打印所有TCP trace样本，保持兼容性"""
        for key in self.rtt_samples:
            print(key, self.rtt_samples[key])

    # 统计相关方法
    def calc_rtt_cdf(self) -> list[float]:
        """计算并返回所有RTT样本的排序列表，用于CDF分析"""
        rtt_list: list[float] = []
        for key in self.rtt_samples:
            for sample in self.rtt_samples[key]:
                rtt_list.append(sample['rtt'])
        rtt_list.sort()
        return rtt_list

    '''
    # 示例用法
    rtt_table = RTTTable()

    # 示例数据包时间戳
    rtt_table.calculate_and_add_rtt('192.168.1.1', '8.8.8.8', 1625140800, 1625140810)
    rtt_table.calculate_and_add_rtt('192.168.1.1', '8.8.8.8', 1625140820, 1625140830)

    # 获取平均RTT
    average_rtt = rtt_table.get_average_rtt('192.168.1.1', '8.8.8.8')
    print(f"Average RTT for 192.168.1.1 to 8.8.8.8: {average_rtt} seconds")

    # 清理旧样本（示例中阈值设为30天）
    rtt_table.clean_old_samples(30 * 24 * 60 * 60)
    '''