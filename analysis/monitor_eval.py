import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import pickle
import math
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RttTable import RTTTable
from src.Monitor import NetworkTrafficMonitor, CompressedIPNode, CompressedIPTrie   
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def read_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
class MonitorEval:
    def __init__(self, current_monitor : NetworkTrafficMonitor, tcptrace_monitor : NetworkTrafficMonitor, baseline_monitor : NetworkTrafficMonitor):
        self.current_monitor = current_monitor
        self.tcptrace_monitor = tcptrace_monitor
        self.baseline_monitor = baseline_monitor
        
    def plot_rtt(data : NetworkTrafficMonitor):
        MonitorEval.plot_monitor_hierachy(data.ipv4_trie)
        MonitorEval.plot_monitor_hierachy(data.ipv6_trie)
    
    # 统计current和tcptrace之间的差异
    def get_data(self):
        '''
        prefix_valid : dict = {
            'ip1' : { 'SYN-ACK' : [(6.566, 1669599831.210783), (6.457, 1669599831.21085), (6.855, 1669599832.414207), (6.697, 1669599832.414321), (6.294, 1669599833.562967), (6.627, 1669599833.563065), (6.459, 1669599840.101774), (6.815, 1669599840.101904)]}
            ...
            }
        prefix_anormal : dict = {
            'ip1' : { 'SYN-ACK' : [(6.566, 1669599831.210783), (6.457, 1669599831.21085), (6.855, 1669599832.414207), (6.697, 1669599832.414321), (6.294, 1669599833.562967), (6.627, 1669599833.563065), (6.459, 1669599840.101774), (6.815, 1669599840.101904)]}
            ...
            }
        '''
        current_valid, current_anormal = MonitorEval.get_monitor_rtt_samples(self.current_monitor)
        tcptrace_valid, tcptrace_anormal = MonitorEval.get_monitor_rtt_samples(self.tcptrace_monitor)
        baseline_valid, baseline_anormal = MonitorEval.get_monitor_rtt_samples(self.baseline_monitor)
        def merge_rtt_within_threshold(data, threshold=0.2):
            '''
            精简RTT数据，将时间戳相隔小于threshold秒的报文合并为一个
            '''
            merged_data = {}
            for ip, records in data.items():
                merged_data[ip] = {}
                for key, key_records in records.items():  # 遍历每个key，例如SYN-ACK、SYN等
                    filtered_records = []
                    prev_timestamp = None

                    for rtt, timestamp in key_records:
                        if prev_timestamp is None or (timestamp - prev_timestamp) > threshold:
                            filtered_records.append((rtt, timestamp))
                            prev_timestamp = timestamp

                    merged_data[ip][key] = filtered_records

            return merged_data

        def plot_rtt(prefix_valid, prefix_anormal, title):
            # 遍历所有的 IP 地址
            for ip, records in prefix_valid.items():
                for key, key_records in records.items():  # 遍历每个key
                    rtts = [item[0] for item in key_records]
                    timestamps = [item[1] for item in key_records]

                    # 绘制有效的 RTT 折线图
                    plt.plot(timestamps, rtts, label=f'{ip} ({key}) valid')

            for ip, records in prefix_anormal.items():
                for key, key_records in records.items():  # 遍历每个key
                    rtts = [item[0] for item in key_records]
                    timestamps = [item[1] for item in key_records]

                    # 绘制异常的 RTT 折线图
                    plt.plot(timestamps, rtts, linestyle='--', label=f'{ip} ({key}) anormal')

            # 添加图表标题和标签
            plt.title(title)
            plt.xlabel('Timestamp')
            plt.ylabel('RTT (ms)')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

        # 过滤掉0.2秒以内的报文，只保留间隔大于0.2秒的报文
        current_valid = merge_rtt_within_threshold(current_valid)
        current_anormal = merge_rtt_within_threshold(current_anormal)
        tcptrace_valid = merge_rtt_within_threshold(tcptrace_valid)
        tcptrace_anormal = merge_rtt_within_threshold(tcptrace_anormal)
        baseline_valid = merge_rtt_within_threshold(baseline_valid)
        baseline_anormal = merge_rtt_within_threshold(baseline_anormal)

        # 绘制图表
        plot_rtt(current_valid, current_anormal, 'Current Monitor RTT')
        plot_rtt(tcptrace_valid, tcptrace_anormal, 'TCPTrace Monitor RTT')
        plot_rtt(baseline_valid, baseline_anormal, 'Baseline Monitor RTT')
    
    @staticmethod
    def compare_monitors(monitor1 : NetworkTrafficMonitor, monitor2 : NetworkTrafficMonitor):
        monitor1_ipv4, monitor1_ipv6 = [], []
        monitor2_ipv4, monitor2_ipv6 = [], []
        
    def get_monitor_rtt_samples(monitor : NetworkTrafficMonitor):
        ipv4_root = monitor.ipv4_trie.root
        ipv6_root = monitor.ipv6_trie.root
        valid_rtt_samples = {}
        anormal_rtt_samples = {}
        child : CompressedIPNode
        def helper(root : CompressedIPNode, valid_rtt_samples : dict, anormal_rtt_samples : dict):
            if (root.network.prefixlen == 32 and root.network.version == 4) or \
                (root.network.prefixlen == 128 and root.network.version == 6):
                valid_rtt_samples[root.network] = root.rtt_records
                anormal_rtt_samples[root.network] = root.anormal_rtt_records
            else:
                for child in root.children.values():
                    helper(child, valid_rtt_samples, anormal_rtt_samples)
        helper(ipv4_root, valid_rtt_samples, anormal_rtt_samples)
        helper(ipv6_root, valid_rtt_samples, anormal_rtt_samples)
        return valid_rtt_samples, anormal_rtt_samples          
                
            
    def plot_monitor_hierachy(data : CompressedIPTrie):
        pass
    
    
def main():
    current_monitor = read_pickle('../data/current_monitor.pkl')
    new_tcp_monitor = read_pickle('../data/new_tcptrace_monitor.pkl')
    monitor_eval = MonitorEval(current_monitor, new_tcp_monitor)
    MonitorEval.plot_rtt(current_monitor)
    MonitorEval.plot_rtt(new_tcp_monitor)
if __name__ == '__main__':
    main()