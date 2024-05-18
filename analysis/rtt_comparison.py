import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RttTable import RTTTable
import argparse
def combine_rtt_table(main : RTTTable, branch : RTTTable):
    for key in branch.rtt_samples:
        if key in main.rtt_samples:
            main.rtt_samples[key].extend(branch.rtt_samples[key])
        else:
            main.rtt_samples[key] = branch.rtt_samples[key]
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
def convert_decimals_to_floats(data):
    """将包含Decimal的列表转换为浮点数列表"""
    return [float(d) for d in data]
class RTTAnalysis:
    def __init__(self, current_table: RTTTable, baseline_table: RTTTable):
        self.current_samples = current_table.rtt_samples
        self.baseline_samples = baseline_table.rtt_samples
        self.ip_dict = {ip: idx for idx, ip in enumerate(set(self.current_samples.keys()) | set(self.baseline_samples.keys()))}
        for key in self.current_samples:
            for value in self.current_samples[key]:
                value['timestamp'] = float(value['timestamp'])
                value['rtt'] = float(value['rtt'])
        for key in self.baseline_samples:
            for value in self.baseline_samples[key]:
                value['timestamp'] = float(value['timestamp'])
                value['rtt'] = float(value['rtt'])
    def calculate_differences(self) -> dict:
        """计算当前数据和基线数据的RTT差异"""
        differences = {}
        for key in self.current_samples:
            if key in self.baseline_samples:
                current_rtts = [sample['rtt'] for sample in self.current_samples[key]]
                baseline_rtts = [sample['rtt'] for sample in self.baseline_samples[key]]
                if current_rtts and baseline_rtts:
                    avg_current_rtt = sum(current_rtts) / len(current_rtts)
                    avg_baseline_rtt = sum(baseline_rtts) / len(baseline_rtts)
                    differences[key] = avg_current_rtt - avg_baseline_rtt
        return differences

    def plot_differences(self) -> None:
        """绘制RTT差异图"""
        differences = self.calculate_differences()
        keys = list(differences.keys())
        values = [differences[key] for key in keys]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(differences)), values, tick_label=[f"{k[0]}-{k[1]}" for k in keys])
        plt.title('Difference in RTT: Current vs Baseline')
        plt.xlabel('IP Pair')
        plt.ylabel('RTT Difference (ms)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_3d_rtt(self):
        """绘制三维RTT图"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 处理当前数据
        for ip_pair, samples in self.current_samples.items():
            x = [self.ip_to_int(ip_pair)] * len(samples)
            y = [sample['timestamp'] for sample in samples]
            z = [sample['rtt'] for sample in samples]
            ax.scatter(x, y, z, color='blue', label='Current RTTs' if ip_pair == list(self.current_samples.keys())[0] else "")
        plt.show()
        # 处理基线数据
        for ip_pair, samples in self.baseline_samples.items():
            x = [self.ip_to_int(ip_pair)] * len(samples)
            y = [sample['timestamp'] for sample in samples]
            z = [sample['rtt'] for sample in samples]
            ax.scatter(x, y, z, color='red', label='Baseline RTTs' if ip_pair == list(self.baseline_samples.keys())[0] else "")
        
        ax.set_xlabel('IP Pair Index')
        ax.set_ylabel('Timestamp')
        ax.set_zlabel('RTT (ms)')
        ax.legend()
        plt.title('3D Plot of RTT Distributions')
        plt.show()

    def ip_to_int(self, ip_pair):
        """将IP地址对转换为唯一的整数，用于图表的x轴。"""
        return self.ip_dict[ip_pair]
    def plot_cdf(self, is_baseline = False):
        rtt_list = self.calc_rtt_cdf(is_baseline)
        n = len(rtt_list)
        y = [(i + 1) / n for i in range(n)]
        plt.plot(rtt_list, y)
        plt.xlabel('RTT (seconds)')
        plt.ylabel('CDF')
        plt.title('RTT CDF')
        plt.show()
    def calc_rtt_cdf(self, is_baseline = False):
        rtt_list = []
        if is_baseline:
            for key in self.baseline_samples:
                for sample in self.baseline_samples[key]:
                    rtt_list.append(sample['rtt'])
        else:
            for key in self.current_samples:
                for sample in self.current_samples[key]:
                    rtt_list.append(sample['rtt'])
        rtt_list.sort()
        return rtt_list
def main(current_files, baseline_file):
    # 加载基线数据
    baseline_table = RTTTable()
    baseline_table.load_rtt_samples(baseline_file)
    
    # 合并当前数据
    current_table = RTTTable()
    for file in current_files:
        current_table.load_rtt_samples(file)

    # 实例化分析类
    rtt_analysis = RTTAnalysis(current_table, baseline_table)
    
    # 执行分析和绘图
    rtt_analysis.plot_3d_rtt()

if __name__ == '__main__':
    
    '''parser = argparse.ArgumentParser(description="Compare RTT data from multiple current files against a single baseline file.")
    parser.add_argument('-b', '--baseline', type=str, required=True, help='The path to the baseline RTT data file.')
    parser.add_argument('-t', '--current', type=str, nargs='+', required=True, help='The path to the current RTT data files.')
    
    args = parser.parse_args()
    main(args.baseline, args.current)'''
    path = './data/rtt'
    for file in os.listdir(path):
        print(file)
    baseline_file = ['icmp_rtt.pkl']
    current_files = ['icmp_dns_ntp_rtt.pkl', 'tcp_rtt.pkl']
    baseline = load_data_from_pickle(os.path.join(path, baseline_file[0]))
    current = RTTTable()
    for file in current_files:
        temp = load_data_from_pickle(os.path.join(path, file))
        combine_rtt_table(current, temp)    
    for key in current.rtt_samples:
        if key in baseline.rtt_samples:
            current.rtt_samples[key].extend(baseline.rtt_samples[key])
    rtt_analysis = RTTAnalysis(current, baseline)
    print(rtt_analysis.current_samples.keys())
    print(rtt_analysis.baseline_samples.keys())
    rtt_analysis.plot_3d_rtt()
    rtt_analysis.plot_cdf(True)
    rtt_analysis.plot_cdf(False)
    