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
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class RelayAnalysis:
    def __init__(self, current_table: RTTTable, baseline_table: RTTTable, tcptrace_table: RTTTable = None):
        '''
        current_table:  current rtt table
        baseline_table: baseline rtt table
        tcptrace_table: tcptrace rtt table
        
        '''
        self.current_samples = copy.deepcopy(current_table.rtt_samples)
        self.baseline_samples = copy.deepcopy(baseline_table.rtt_samples)
        self.tcptrace_samples = copy.deepcopy(tcptrace_table.rtt_samples) if tcptrace_table else None
        self.current_rtt, self.baseline_rtt, self.tcptrace_rtt = {}, {}, {}
        
        self.ip_dict = {ip: idx for idx, ip in enumerate(set(self.current_samples.keys()) | set(self.baseline_samples.keys()))}
        for key in self.current_samples:
            for value in self.current_samples[key]:
                value['timestamp'] = float(value['timestamp'])
                value['rtt'] = float(value['rtt']) * 1000
                if value['rtt'] > 1000:
                    print(value, end='\n')
        for key in self.baseline_samples:
            for value in self.baseline_samples[key]:
                value['timestamp'] = float(value['timestamp'])
                value['rtt'] = float(value['rtt']) * 1000
    def get_all_samples(self):
        '''
        183.173.245.106-117.176.244.18
        RTT: 0.040275 seconds, Timestamp: 1714302542.445801, types: PSH, direction: backward
        RTT: 0.000158 seconds, Timestamp: 1714302542.445961, types: Normal, direction: forward
        RTT: 0.000111 seconds, Timestamp: 1714302542.447062, types: Normal, direction: forward
        RTT: 0.042834 seconds, Timestamp: 1714302554.422473, types: PSH, direction: backward
        RTT: 0.000128 seconds, Timestamp: 1714302554.423946, types: Normal, direction: forward
        RTT: 0.000129 seconds, Timestamp: 1714302554.423947, types: PSH, direction: forward
        RTT: 0.039844 seconds, Timestamp: 1714302599.607222, types: PSH, direction: backward
        RTT: 0.038955 seconds, Timestamp: 1714302599.607224, types: PSH, direction: backward
                
        '''
        for key in self.current_samples:
            src_ip, dst_ip = key.split('-')
            for value in self.current_samples[key]:
                if value['direction'] == 'forward':
                    if src_ip not in self.current_rtt:
                        self.current_rtt[src_ip] = []
                    self.current_rtt[src_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
                else:
                    if dst_ip not in self.current_rtt:
                        self.current_rtt[dst_ip] = []
                    self.current_rtt[dst_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
        for key in self.baseline_samples:
            src_ip, dst_ip = key.split('-')
            for value in self.baseline_samples[key]:
                if value['direction'] == 'forward':
                    if src_ip not in self.baseline_rtt:
                        self.baseline_rtt[src_ip] = []
                    self.baseline_rtt[src_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
                else:
                    if dst_ip not in self.baseline_rtt:
                        self.baseline_rtt[dst_ip] = []
                    self.baseline_rtt[dst_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
        if self.tcptrace_samples:
            for key in self.tcptrace_samples:
                src_ip, dst_ip = key.split('-')
                for value in self.tcptrace_samples[key]:
                    if value['direction'] == 'forward':
                        if src_ip not in self.tcptrace_rtt:
                            self.tcptrace_rtt[src_ip] = []
                        self.tcptrace_rtt[src_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
                    else:
                        if dst_ip not in self.tcptrace_rtt:
                            self.tcptrace_rtt[dst_ip] = []
                        self.tcptrace_rtt[dst_ip].append({'rtt': value['rtt'], 'timestamp': value['timestamp'], 'types': value['types']})
    def print_rtt_samples(self, folder):
        os.makedirs(folder, exist_ok=True)  # exist_ok=True 防止抛出异常如果目录已存在
        with open(folder + '/current.txt', 'w') as file:
            for key in self.current_samples:
                for value in self.current_samples[key]:
                    print(key, value, end='\n', file=file)
        with open(folder + '/baseline.txt', 'w') as file:
            for key in self.baseline_samples:
                for value in self.baseline_samples[key]:
                    print(key, value, end='\n',file=file)
        with open(folder + '/tcptrace.txt', 'w') as file:
            for key in self.tcptrace_samples:
                for value in self.tcptrace_samples[key]:
                    print(key, value, end='\n',file=file)
    def plot_every_ip(self):
        for ip in self.tcptrace_rtt :
            if ip in self.baseline_rtt:
                for value in self.baseline_rtt[ip]:
                    plt.plot(value['timestamp'], value['rtt'], 'r*', markersize = 3)
            if ip in self.current_rtt:
                for value in self.current_rtt[ip]:
                    plt.plot(value['timestamp'], value['rtt'], 'bx', markersize = 4)
            for value in self.tcptrace_rtt[ip]:
                plt.plot(value['timestamp'], value['rtt'], 'go', markersize = 3)
            plt.title(ip)
            plt.show()
    def print_rtt(self):
        output_dir = 'data/output3/'
        os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 防止抛出异常如果目录已存在
    
        with open (output_dir + 'current_rtt.txt', 'w') as file:
            for key in self.current_rtt:
                print(key, self.current_rtt[key], end='\n', file=file)
        with open (output_dir + 'baseline_rtt.txt', 'w') as file:
            for key in self.baseline_rtt:
                print(key, self.baseline_rtt[key], end='\n', file=file)
        with open (output_dir + 'tcptrace_rtt.txt', 'w') as file:
            for key in self.tcptrace_rtt:
                print(key, self.tcptrace_rtt[key], end='\n', file=file)
    def calculate_average_rtt(self, rtt_data, timeslot_size, is_outlier = False):
        """
        计算指定 RTT 数据中每个 IP 的平均 RTT，按时间槽分组。

        :param rtt_data: RTT 数据字典，键是 IP 地址，值是 RTT 记录列表。
        :param timeslot_size: 时间槽大小，单位是秒。
        :return: 每个 IP 地址的时间槽的平均 RTT 的字典。
        """

        if not is_outlier:
            rtt_by_timeslot = {}
            for ip, entries in rtt_data.items():
                if ip not in rtt_by_timeslot:
                    rtt_by_timeslot[ip] = {}
                for entry in entries:
                    timeslot = round(math.floor(entry['timestamp'] / timeslot_size) * timeslot_size,4)
                    if timeslot not in rtt_by_timeslot[ip]:
                        rtt_by_timeslot[ip][timeslot] = []
                    rtt_by_timeslot[ip][timeslot].append(entry['rtt'])
            average_rtt_per_ip_timeslot = {}
            for ip, timeslots in rtt_by_timeslot.items():
                average_rtt_per_timeslot = {}
                for timeslot, rtts in timeslots.items():
                    average_rtt = sum(rtts) / len(rtts) if rtts else None
                    average_rtt_per_timeslot[timeslot] = average_rtt
                average_rtt_per_ip_timeslot[ip] = average_rtt_per_timeslot
            return average_rtt_per_ip_timeslot, rtt_by_timeslot
        else:
            rtt_by_timeslot = {}
            for ip, entries in rtt_data.items():
                if ip not in rtt_by_timeslot:
                    rtt_by_timeslot[ip] = {}
                for entry in entries:
                    timeslot = round(math.floor(entry['timestamp'] / timeslot_size) * timeslot_size, 4)
                    if timeslot not in rtt_by_timeslot[ip]:
                        rtt_by_timeslot[ip][timeslot] = []
                    rtt_by_timeslot[ip][timeslot].append(entry['rtt'])

            average_rtt_per_ip_timeslot = {}
            for ip, timeslots in rtt_by_timeslot.items():
                average_rtt_per_timeslot = {}
                for timeslot, rtts in timeslots.items():
                    rtts_filtered = remove_outliers(rtts)
                    average_rtt = sum(rtts_filtered) / len(rtts_filtered) if rtts_filtered else None
                    average_rtt_per_timeslot[timeslot] = average_rtt
                average_rtt_per_ip_timeslot[ip] = average_rtt_per_timeslot
            return average_rtt_per_ip_timeslot, rtt_by_timeslot
    def process_all_rtt_data(self, timeslot_size = 0.2):
        """
        处理所有 RTT 数据并存储每个字典中每个 IP 的平均 RTT 结果到相应属性。
        :param timeslot_size: 时间槽大小，单位是秒。
        """
        self.current_timeslot, self.current_timeslot_samples = self.calculate_average_rtt(self.current_rtt, timeslot_size, is_outlier = True)
        self.baseline_timeslot, self.baseline_timeslot_samples = self.calculate_average_rtt(self.baseline_rtt, timeslot_size)
        self.tcptrace_timeslot, self.tcptrace_timeslot_samples = self.calculate_average_rtt(self.tcptrace_rtt, timeslot_size)
    def print_rtt_by_timeslot(self, folder):
        # 输出 current_timeslot
        with open(folder + '/current_timeslot.txt', 'w') as file:
            for key in sorted(self.current_timeslot):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.current_timeslot[key]):
                    print(key2, end=' : ', file=file)
                    print(self.current_timeslot[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)

        # 输出 baseline_timeslot
        with open(folder + '/baseline_timeslot.txt', 'w') as file:
            for key in sorted(self.baseline_timeslot):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.baseline_timeslot[key]):
                    print(key2, end=' : ', file=file)
                    print(self.baseline_timeslot[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)

        # 输出 tcptrace_timeslot
        with open(folder + '/tcptrace_timeslot.txt', 'w') as file:    
            for key in sorted(self.tcptrace_timeslot):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.tcptrace_timeslot[key]):
                    print(key2, end=' : ', file=file)
                    print(self.tcptrace_timeslot[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)
        # 输出 current_timeslot_samples
        with open(folder + '/current_timeslot_samples.txt', 'w') as file:
            for key in sorted(self.current_timeslot_samples):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.current_timeslot_samples[key]):
                    print(key2, end=' : ', file=file)
                    print(self.current_timeslot_samples[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)

        # 输出 baseline_timeslot_samples
        with open(folder + '/baseline_timeslot_samples.txt', 'w') as file:
            for key in sorted(self.baseline_timeslot_samples):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.baseline_timeslot_samples[key]):
                    print(key2, end=' : ', file=file)
                    print(self.baseline_timeslot_samples[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)

        # 输出 tcptrace_timeslot_samples
        with open(folder + '/tcptrace_timeslot_samples.txt', 'w') as file:
            for key in sorted(self.tcptrace_timeslot_samples):
                print('-----------------------------------', end='\n', file=file)
                print(key, end='\n', file=file)
                for key2 in sorted(self.tcptrace_timeslot_samples[key]):
                    print(key2, end=' : ', file=file)
                    print(self.tcptrace_timeslot_samples[key][key2], end='\n', file=file)
                print('***********************************', end='\n', file=file)
    def plot_all_ip(self):
        
        for ip in self.tcptrace_timeslot:
            if ip in self.baseline_timeslot:
                plt.plot(self.baseline_timeslot[ip].keys(), self.baseline_timeslot[ip].values(), 'r', label='Baseline')
            if ip in self.current_timeslot:
                plt.plot(self.current_timeslot[ip].keys(), self.current_timeslot[ip].values(), 'b', label='Current')
            if ip in self.tcptrace_timeslot:
                plt.plot(self.tcptrace_timeslot[ip].keys(), self.tcptrace_timeslot[ip].values(), 'g', label='TCPtrace')
            plt.title(ip)
            plt.show()       
    def plot_all_ip1(self):
        # 遍历所有 IP 地址
        for ip in self.tcptrace_timeslot:
            plt.figure(figsize=(10, 5))  # 设置图形的尺寸
            # 对于存在于基线数据中的 IP
            if ip in self.baseline_timeslot:
                # 获取时间戳并按升序排序
                sorted_keys = sorted(self.baseline_timeslot[ip])
                sorted_values = [self.baseline_timeslot[ip][key] for key in sorted_keys]
                plt.plot(sorted_keys, sorted_values, 'r', label='Baseline')
            
            # 对于存在于当前数据中的 IP
            if ip in self.current_timeslot:
                sorted_keys = sorted(self.current_timeslot[ip])
                sorted_values = [self.current_timeslot[ip][key] for key in sorted_keys]
                plt.plot(sorted_keys, sorted_values, 'b', label='Current')
            
            # 对于存在于 TCPtrace 数据中的 IP
            if ip in self.tcptrace_timeslot:
                sorted_keys = sorted(self.tcptrace_timeslot[ip])
                sorted_values = [self.tcptrace_timeslot[ip][key] for key in sorted_keys]
                plt.plot(sorted_keys, sorted_values, 'g', label='TCPtrace')
            
            plt.title(f"RTT Data for IP: {ip}")
            plt.xlabel('Timestamp')
            plt.ylabel('RTT')
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格
            plt.show()
    def plot_everytype_current(self):
        dns_dict, icmp_dict, tcp_dict = {}, {}, {}
        for key in self.current_samples:
            ip1, ip2 = key.split('-')
            for value in self.current_samples[key]:
                if value['direction'] == 'forward':
                    if value['types'] == 'DNS':
                        if ip1 not in dns_dict:
                            dns_dict[ip1] = []
                        dns_dict[ip1].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})
                    elif value['types'] == 'Normal' or value['types'] == 'PSH':
                        if ip1 not in tcp_dict:
                            tcp_dict[ip1] = []
                        tcp_dict[ip1].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})
                else:
                    if value['types'] == 'DNS':
                        if ip2 not in dns_dict:
                            dns_dict[ip2] = []
                        dns_dict[ip2].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})
                    elif value['types'] == 'Normal' or value['types'] == 'PSH':
                        if ip2 not in tcp_dict:
                            tcp_dict[ip2] = []
                        tcp_dict[ip2].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})
        for key in self.baseline_samples:
            ip1, ip2 = key.split('-')
            for value in self.baseline_samples[key]:
                if value['direction'] == 'forward':
                    if ip1 not in icmp_dict:
                        icmp_dict[ip1] = []
                    icmp_dict[ip1].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})
                else:
                    if ip2 not in icmp_dict:
                        icmp_dict[ip2] = []
                    icmp_dict[ip2].append({'rtt': value['rtt'], 'timestamp': value['timestamp']})

        for ip in dns_dict:
            plt.plot([value['timestamp'] for value in dns_dict[ip]], [value['rtt'] for value in dns_dict[ip]], 'r*', markersize = 5)
            if ip in tcp_dict:
                plt.plot([value['timestamp'] for value in tcp_dict[ip]], [value['rtt'] for value in tcp_dict[ip]], 'bx', markersize = 5)
            if ip in icmp_dict:
                plt.plot([value['timestamp'] for value in icmp_dict[ip]], [value['rtt'] for value in icmp_dict[ip]], 'go', markersize = 5)
            plt.title(ip)
            plt.grid(True)
            
            plt.show()
                    
def remove_outliers(data):
    """
    去除数据中的离群值，使用更宽松的标准。

    :param data: 数据列表。
    :return: 清除离群值后的数据列表。
    """
    if len(data) < 4:
        return data  # 数据点太少，不去除离群值
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    # 使用 3 倍的 IQR 来确定离群值的边界
    lower_bound = quartile_1 - 1.5 * iqr
    upper_bound = quartile_3 + 0 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]
def sort_data_by_timestamp(data):
    sorted_data = {}
    for ip, entries in data.items():
        # 排序每个 IP 的记录，按照 timestamp 字段
        sorted_entries = sorted(entries, key=lambda x: x['timestamp'])
        sorted_data[ip] = sorted_entries
    return sorted_data
def calculate_overlap(tcptrace_timeslot, other_timeslot):
    """计算覆盖率，即两个时间槽字典的重合率。"""
    tcptrace_keys = set(tcptrace_timeslot.keys())
    other_keys = set(other_timeslot.keys())
    intersection = tcptrace_keys.intersection(other_keys)
    if not tcptrace_keys:  # 避免除以零
        return 0
    return len(intersection) / len(tcptrace_keys)

def calculate_relative_error(tcptrace_timeslot, current_timeslot, baseline_timeslot):
    """计算相对于baseline的误差。"""
    errors = {}
    for ip, baseline_data in baseline_timeslot.items():
        if ip in tcptrace_timeslot and ip in current_timeslot:
            errors[ip] = {}
            for timeslot in baseline_data.keys():
                if timeslot in tcptrace_timeslot[ip] and timeslot in current_timeslot[ip]:
                    baseline_rtt = baseline_data[timeslot]
                    tcptrace_error = abs(tcptrace_timeslot[ip][timeslot] - baseline_rtt)
                    current_error = abs(current_timeslot[ip][timeslot] - baseline_rtt)
                    errors[ip][timeslot] = {'tcptrace_error': tcptrace_error, 'current_error': current_error}
    return errors

# 假设数据

                
                    
def load_data_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except pickle.UnpicklingError:
        print(f"Error: The data in {file_path} could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
def combine_rtt_table(main : RTTTable, branch : RTTTable):
    for key in branch.rtt_samples:
        if key in main.rtt_samples:
            main.rtt_samples[key].extend(branch.rtt_samples[key])
        else:
            main.rtt_samples[key] = branch.rtt_samples[key]
def calculate_inner_multiply(lst1, lst2):
    inner_multiply = sum(x * y for x, y in zip(lst1, lst2))
    return inner_multiply
def calculate_sum_of_list(lst):
    sum_of_list = sum(lst)
    return sum_of_list
def process_tcptrace_data(tcptrace_data: RTTTable):
    count = 0
    '''
    tcptrace data:
    key :   183.173.245.106-101.70.156.51 
    value   :   [
        {'rtt_samples': [1, 0], 'first_packet': 1714302717.0, 'last_packet': 1714302717.0, 'rtt_min': [0.3, 0.0], 'rtt_max': [0.3, 0.0], 'rtt_avg': [0.3, 0.0], 'port_a': '443', 'port_b': '56445'}, 
        
        {'rtt_samples': [14, 5], 'first_packet': 1714302711.0, 'last_packet': 1714302712.0, 'rtt_min': [54.4, 0.2], 'rtt_max': [61.6, 0.9], 'rtt_avg': [56.5, 0.5], 'port_a': '56445', 'port_b': '443'}
        ]
    '''
    tcptrace_rtttable = RTTTable()
    data = tcptrace_data.rtt_samples
    
    
    for key in data:
        for value in data[key]:
            temp = {}
            temp['timestamp'] = value['first_packet']
            if calculate_sum_of_list(value['rtt_samples']) == 0:
                continue
            temp['rtt'] = calculate_inner_multiply(value['rtt_samples'], value['rtt_avg']) / calculate_sum_of_list(value['rtt_samples'])
            #
            count += calculate_sum_of_list(value['rtt_samples'])
            temp['rtt_samples'] = value['rtt_samples']
            tcptrace_rtttable.add_rtt_sample(key.split('-')[0], key.split('-')[1], rtt=temp['rtt'], timestamp=temp['timestamp'], types='tcptrace')
    print(count, '**')
    return tcptrace_rtttable
def process_min_tcptrace_data(tcptrace_data: RTTTable):
    count = 0
    '''
    tcptrace data:
    key :
    value   :
    RTT: 0.120163 ms, Timestamp: 1714302628.902181, types: tcptrace, direction: backward
    RTT: 26.799917 ms, Timestamp: 1714302633.313756, types: tcptrace, direction: forward
    RTT: 0.067949 ms, Timestamp: 1714302633.313756, types: tcptrace, direction: backward
    '''
    tcptrace_rtttable = RTTTable()
    data = tcptrace_data.rtt_samples
    for key in data:
        for value in data[key]:
            temp = {}
            temp['timestamp'] = value['first_packet']
            if calculate_sum_of_list(value['rtt_samples']) == 0:
                continue
            temp['rtt'] = min(value['rtt_min'])
            #
            temp['rtt_samples'] = value['rtt_samples']
            tcptrace_rtttable.add_rtt_sample(key.split('-')[0], key.split('-')[1], rtt=temp['rtt'], timestamp=temp['timestamp'], types='tcptrace')
    return tcptrace_rtttable
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare RTT data from multiple sources.")
    parser.add_argument('rtt_pkl', type=str, default='./data/rtt5', help='Path to the RTT data files.')
    parser.add_argument('tcptrace_pkl', type=str, default='./data/tcptrace_table/new_tcptrace.pkl', help='Path to the tcptrace pickle file.')
    parser.add_argument('output_folder', type=str, default='./output', help='Folder to save the output files.')
    parser.add_argument('--timeslot', type=float, default=0.2, help='Size of the timeslot in seconds.')
    
    args = parser.parse_args()
    path = args.rtt_pkl
    tcptrace_pkl = args.tcptrace_pkl+'/new_tcptrace.pkl'
    folder = args.output_folder
    timeslot = args.timeslot
    # 打印出文件夹中的文件
    for file in os.listdir(path):
        print(file)
    
    # 加载基线文件
    baseline_file = 'icmp_rtt.pkl'
    baseline = load_data_from_pickle(os.path.join(path, baseline_file))
    
    # 创建并组合当前RTT表
    current = RTTTable()
    '''    current_files = ['icmp_dns_ntp_rtt.pkl', 'tcp_rtt.pkl']
    for file in current_files:
        temp = load_data_from_pickle(os.path.join(path, file))
        combine_rtt_table(current, temp)'''
    current = load_data_from_pickle(os.path.join(path, 'tcp_rtt.pkl'))
    # 加载tcptrace数据
    tcptrace = load_data_from_pickle(tcptrace_pkl)
    
    # 创建分析对象并进行分析
    relayanal = RelayAnalysis(current, baseline, tcptrace)
    relayanal.get_all_samples()
    relayanal.print_rtt_samples(folder)
    #relayanal.plot_every_ip()
    relayanal.print_rtt()
    relayanal.process_all_rtt_data(timeslot_size=timeslot)
    relayanal.print_rtt_by_timeslot(folder)
    #relayanal.plot_everytype_current()
    #relayanal.plot_all_ip1()
    
    # 计算并打印统计信息
    tcptrace_current_overlap = calculate_overlap(relayanal.tcptrace_timeslot, relayanal.current_timeslot)
    tcptrace_baseline_overlap = calculate_overlap(relayanal.tcptrace_timeslot, relayanal.baseline_timeslot)
    relative_errors = calculate_relative_error(relayanal.tcptrace_timeslot, relayanal.current_timeslot, relayanal.baseline_timeslot)
    
    print(f"TCPtrace 和 Current 的覆盖率: {tcptrace_current_overlap:.2%}")
    print(f"TCPtrace 和 Baseline 的覆盖率: {tcptrace_baseline_overlap:.2%}")
    print("相对于 Baseline 的误差：")
    for ip, data in relative_errors.items():
        print(f"IP: {ip}")
        for timeslot, errors in data.items():
            print(f"  Timeslot: {timeslot}, TCPtrace Error: {errors['tcptrace_error']}, Current Error: {errors['current_error']}")

    for ip, data in relative_errors.items():
        if data:
            print(f"IP: {ip}")
            print('average tcptrace error:', sum([errors['tcptrace_error'] for timeslot, errors in data.items()]) / len(data))
            print('average current error:', sum([errors['current_error'] for timeslot, errors in data.items()]) / len(data))

if __name__ == '__main__':
    main()

# timeslot 0.2s
#python3 ./analysis/rtt_relay.py ./current/result3  ./tcptrace/tcpresult3 output/test3 --timeslot=0.05