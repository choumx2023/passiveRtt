import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import pickle
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RttTable import RTTTable
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
            temp['rtt'] = min(value['rtt_min'])
            #
            temp['rtt_samples'] = value['rtt_samples']
            tcptrace_rtttable.add_rtt_sample(key.split('-')[0], key.split('-')[1], rtt=temp['rtt'], timestamp=temp['timestamp'], types='tcptrace')   
    return tcptrace_rtttable
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
    def __init__(self, current_table: RTTTable, baseline_table: RTTTable, tcptrace_table: RTTTable = None):
        '''
        current_table:  current rtt table
        baseline_table: baseline rtt table
        tcptrace_table: tcptrace rtt table
        
        '''
        self.current_samples = copy.deepcopy(current_table.rtt_samples)
        self.baseline_samples = copy.deepcopy(baseline_table.rtt_samples)
        self.tcptrace_samples = copy.deepcopy(tcptrace_table.rtt_samples) if tcptrace_table else None
        self.ip_dict = {ip: idx for idx, ip in enumerate(set(self.current_samples.keys()) | set(self.baseline_samples.keys()))}
    def second_to_millisecond(self, second):
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
    def check(self):
        for key in self.current_samples:
            for value in self.current_samples[key]:
                if value['rtt'] > 1000:
                    print(value, end='\n')
        for key in self.baseline_samples:
            for value in self.baseline_samples[key]:
                if value['rtt'] > 1000:
                    print(value, end='\n')
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

    def plot_3d_rtt(self, rtt_min=0, rtt_max=1500):
        """绘制三维RTT图，只包括特定RTT区间的数据"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 处理当前数据
        for ip_pair, samples in self.current_samples.items():
            filtered_samples = [s for s in samples if rtt_min <= s['rtt'] <= rtt_max]
            if filtered_samples:  # 确保有数据才进行绘图
                x = [self.ip_dict[ip_pair]] * len(filtered_samples)
                y = [s['timestamp'] for s in filtered_samples]
                z = [s['rtt'] for s in filtered_samples]
                ax.plot3D(x, y, z, color='blue', label='Current RTTs' if ip_pair == list(self.current_samples.keys())[0] else "")

        # 处理基线数据
        
        for ip_pair, samples in self.baseline_samples.items():
            filtered_samples = [s for s in samples if rtt_min <= s['rtt'] <= rtt_max]
            if filtered_samples:  # 确保有数据才进行绘图
                x = [self.ip_dict[ip_pair]] * len(filtered_samples)
                y = [s['timestamp'] for s in filtered_samples]
                z = [s['rtt'] for s in filtered_samples]
                ax.plot3D(x, y, z, color='red', label='Baseline RTTs' if ip_pair == list(self.baseline_samples.keys())[0] else "")

        ax.set_xticks(range(len(self.ip_dict)))
        ax.set_xlabel('IP Index')
        ax.set_ylabel('Timestamp')
        ax.set_zlabel('RTT (ms)')
        ax.set_title('RTT Comparison: Current vs Baseline (Filtered)')
        plt.legend()
        plt.show()

        
        plt.show()
    def plot_cdf(self):
        rtt_list1 = self.calc_rtt_cdf(True)
        n1 = len(rtt_list1)
        y1 = [(i + 1) / n1 for i in range(n1)]
        print(n1, len(rtt_list1))
        print(rtt_list1[n1 - 1])
        plt.plot(rtt_list1, y1, label='Baseline RTT', color='blue')
        
        rtt_list2 = self.calc_rtt_cdf(False)
        n2 = len(rtt_list2)
        y2 = [(i + 1) / n2 for i in range(n2)]
        print(rtt_list2[n2 - 1])
        print(n2, len(rtt_list2))
        plt.plot(rtt_list2, y2, label='Current RTT', color='red')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 150])
        plt.xlabel('RTT (ms)')
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
    def plot_rtt_by_ipaddress(self):
        for ip in self.ip_dict:
            print(f"Processing IP: {ip}")  # 打印当前处理的IP
            
            rtt_list1 = [sample['rtt'] for key in self.baseline_samples if key[0] == ip for sample in self.baseline_samples[key] if sample['rtt'] <= 1000]
            rtt_list1.sort()
            print(f"Baseline RTT List for {ip}: {rtt_list1}")  # 查看列表内容
            
            if rtt_list1:
                n1 = len(rtt_list1)
                y1 = [(i + 1) / n1 for i in range(n1)]
                plt.plot(rtt_list1, y1, label=f'Baseline RTT {ip}', color='blue')

            rtt_list2 = [sample['rtt'] for key in self.current_samples if key[0] == ip for sample in self.current_samples[key] if sample['rtt'] <= 1000]
            rtt_list2.sort()
            print(f"Current RTT List for {ip}: {rtt_list2}")  # 查看列表内容

            if rtt_list2:
                n2 = len(rtt_list2)
                y2 = [(i + 1) / n2 for i in range(n2)]
                plt.plot(rtt_list2, y2, label=f'Current RTT {ip}', color='red')

            if rtt_list1 or rtt_list2:
                plt.legend()
                plt.xlabel('RTT (ms)')
                plt.ylabel('CDF')
                plt.title(f'RTT CDF for IP {ip}')
                plt.show()
            else:
                print(f"No data to plot for IP {ip}")

            
    def calc_current_cover(self):
        pass
    def compare_with_tcptrce(self, tcptrace_table: RTTTable):
        start_time = tcptrace_table.min_time
        end_time = tcptrace_table.max_time
        result = {}
        result['start_time'] = start_time
        for key in tcptrace_table.rtt_samples:
            result[key] = [tcptrace_table.rtt_samples[key], [], []]
            if key in self.current_samples:
                for sample in current.rtt_samples[key]:
                    if start_time - 1 <= sample['timestamp'] <= end_time + 1:
                        result[key][1].append(sample)
            if key in self.baseline_samples:
                for sample in baseline.rtt_samples[key]:
                    if start_time - 1 <= sample['timestamp'] <= end_time + 1:
                        result[key][2].append(sample)
        '''
        result = [
            tcptrace:  
            current :     
            baseline: 
            
            
            
        ]
        '''
        return result
    def print_result(self, output_file):
        with open(output_file, 'w') as file:
            for key in self.current_samples:
                for value in self.current_samples[key]:
                    print(key, value, end='\n', file=file)
            for key in self.baseline_samples:
                for value in self.baseline_samples[key]:
                    print(key, value, end='\n',file=file)
    
    def agg_tcptrace_by_ipaddress(self, tcptrace_dir):
        tcptrace_data = {}
        for tcptrace_file in tcptrace_dir:
            temp = load_data_from_pickle(tcptrace_file)
            temp : RTTTable
            for key in temp.rtt_samples:
                tcptrace_data[key] = []
                values = temp.rtt_samples[key]
                for value in values:
                    min_rtt = min(min_rtt, value['rtt'])
                    rtt_samples = value['rtt_samples']
    def plot_every_ipaddress(self):
        count = 0   
        for key in self.baseline_samples:
            count += 1
            for value in self.baseline_samples[key]:
                plt.plot(value['timestamp'], value['rtt'], 'r*', markersize= 6)
            if key in self.current_samples:
                for value in self.current_samples[key]:
                    plt.plot(value['timestamp'], value['rtt'], 'bo', markersize=2)
            if key in self.tcptrace_samples:
                for value in self.tcptrace_samples[key]:
                    plt.plot(value['timestamp'], value['rtt'], 'go',markersize=2)
            plt.title(f'RTT for IP Pair {key}')
            plt.show()
    def plot_every_ipaddress_samples_number(self):
        count = 0
        total_baseline, total_current, total_tcptrace = 0, 0, 0
        result = []
        for key in self.baseline_samples:
            count += 1
            baseline_num = len(self.baseline_samples[key])
            current_num = len(self.current_samples[key]) if key in self.current_samples else 0
            tcptrace_num = len(self.tcptrace_samples[key]) if key in self.tcptrace_samples else 0
            total_baseline += baseline_num
            total_current += current_num
            total_tcptrace += tcptrace_num
            
            result.append([baseline_num, current_num, tcptrace_num])
            
        plt.plot(result)
        plt.ylim([0, 600])
        plt.legend(['Baseline', 'Current', 'Tcptrace'])#11808 949 6006
        plt.show()
        print(total_baseline, total_current, total_tcptrace)
    def print_rtt_samples(self, folder):
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
    def calculate_data_coverage(self, based_on='tcptrace'):
        '''
        result_current: {
            ip_pair: [coverage, number of samples]
        }
        '''
        result_current = {}
        tcptrace_count = 0
        current_count = 0
        baseline_count = 0
        less_ip = set()
        coverage = 0
        output_file = 'data/output/'
        if based_on == 'tcptrace':
            # 计算当前数据和tcptrace的覆盖率
            print('len:', len(self.tcptrace_samples))
            for key in self.tcptrace_samples:
                if key in self.current_samples:
                    coverage = self.calculate_relative_coverage(self.tcptrace_samples[key], self.current_samples[key])
                    result_current[key] = coverage
                    current_count += len(self.current_samples[key])
                else:
                    result_current[key] = 0
                tcptrace_count += len(self.tcptrace_samples[key])
                if coverage < 0.7:
                    less_ip.add((key, coverage, len(self.tcptrace_samples[key])))
            
            result_baseline = {}
            # 计算基线数据和tcptrace的覆盖率
            for key in self.tcptrace_samples:
                if key in self.baseline_samples:
                    coverage = self.calculate_relative_coverage(self.tcptrace_samples[key], self.baseline_samples[key])
                    result_baseline[key] = coverage
                    baseline_count += len(self.baseline_samples[key])
                else:
                    result_baseline[key] = 0
            print(len(less_ip))
            print(tcptrace_count, current_count, baseline_count)
            for key in self.tcptrace_samples:
                mean_current_coverage = sum(result_current.values()) / len(result_current)
                mean_baseline_coverage = sum(result_baseline.values()) / len(result_baseline)
            print(mean_current_coverage, mean_baseline_coverage)
            
            with open(output_file+'tcpt.txt', 'w') as file:
                for ip in less_ip:
                    print(ip, file=file, end='\n')
            return result_current, result_baseline
        
        else:# based on baseline
            print('len:', len(self.baseline_samples))
            for key in self.baseline_samples:
                if key in self.current_samples:
                    coverage = self.calculate_relative_coverage(self.baseline_samples[key], self.current_samples[key])
                    result_current[key] = coverage
                    current_count += len(self.current_samples[key])
                else:
                    result_current[key] = 0
                baseline_count += len(self.baseline_samples[key])
                if coverage < 0.7:
                    less_ip.add((key, coverage, len(self.baseline_samples[key])))
            result_tcptrace = {}
            for key in self.baseline_samples:
                if key in self.tcptrace_samples:
                    coverage = self.calculate_relative_coverage(self.baseline_samples[key], self.tcptrace_samples[key])
                    result_tcptrace[key] = coverage
                    tcptrace_count += len(self.tcptrace_samples[key])
                else:
                    result_tcptrace[key] = 0
            print(tcptrace_count, current_count, baseline_count)
            mean_current_coverage = sum(result_current.values()) / len(result_current)
            mean_tcptrace_coverage = sum(result_tcptrace.values()) / len(result_tcptrace)
        
            print(mean_current_coverage, mean_tcptrace_coverage)
            with open(output_file+'bl.txt', 'w') as file:
                for ip in less_ip:
                    print(ip, file=file, end='\n')
            print(len(less_ip))
            return result_current, result_tcptrace
        
    import matplotlib.pyplot as plt
    def plot_coverage(self):
        result_current, result_baseline = self.calculate_data_coverage()    
        # 准备绘图数据
        ips = list(result_current.keys())  # 获取所有IP地址对
        current_coverages = [result_current[ip] for ip in ips]  # 获取当前数据集的覆盖率
        baseline_coverages = [result_baseline[ip] for ip in ips]  # 获取基线数据集的覆盖率
        # 设置绘图大小
        plt.figure(figsize=(10, 8))
        # 生成X轴的位置数组
        x = range(len(ips))
        # 绘制条形图
        plt.bar(x, current_coverages, width=0.4, label='Current', align='center')
        plt.bar([i + 0.4 for i in x], baseline_coverages, width=0.4, label='Baseline', align='center')
        # 设置坐标轴标签
        plt.xlabel('IP Address Pairs')
        plt.ylabel('Coverage Percentage')
        # 设置X轴的刻度标签
        plt.xticks([i + 0.2 for i in x], ips, rotation=45, ha="right")  # 将标签旋转以更好地显示
        # 添加图例
        plt.legend()
        # 显示图表
        plt.tight_layout()
        plt.show()

        result_current, result_tcptrace = self.calculate_data_coverage(based_on='baseline')
        # 准备绘图数据
        ips = list(result_current.keys())  # 获取所有IP地址对
        current_coverages = [result_current[ip] for ip in ips]  # 获取当前数据集的覆盖率
        tcptrace_coverages = [result_tcptrace[ip] for ip in ips]  # 获取基线数据集的覆盖率
        # 设置绘图大小
        plt.figure(figsize=(10, 8))
        # 生成X轴的位置数组
        x = range(len(ips))
        # 绘制条形图
        plt.bar(x, current_coverages, width=0.4, label='Current', align='center')
        plt.bar([i + 0.4 for i in x], tcptrace_coverages, width=0.4, label='Tcptrace', align='center')
        # 设置坐标轴标签
        plt.xlabel('IP Address Pairs')
        plt.ylabel('Coverage Percentage')
        # 设置X轴的刻度标签
        plt.xticks([i + 0.2 for i in x], ips, rotation=45, ha="right")  # 将标签旋转以更好地显示
        # 添加图例
        plt.legend()
        # 显示图表
        plt.tight_layout()
        plt.show()
    @staticmethod
    def calculate_relative_coverage(set1, set2):
        seconds_set1 = set()
        seconds_set2 = set()
        for entry in set1:
            second = int(entry['timestamp'])
            seconds_set1.add(second)
        for entry in set2:
            second = int(entry['timestamp'])
            seconds_set2.add(second)
        intersection = seconds_set1.intersection(seconds_set2)  # 修改这里，使用正确的集合
        coverage = (len(intersection) / len(seconds_set1))  if seconds_set1 else 0
        return coverage
            
                    
if __name__ == '__main__':
    
    '''parser = argparse.ArgumentParser(description="Compare RTT data from multiple current files against a single baseline file.")
    parser.add_argument('-b', '--baseline', type=str, required=True, help='The path to the baseline RTT data file.')
    parser.add_argument('-t', '--current', type=str, nargs='+', required=True, help='The path to the current RTT data files.')
    
    args = parser.parse_args()
    main(args.baseline, args.current)'''
    path = './data/rtt5'
    folder = './data/output1'
    for file in os.listdir(path):
        print(file)
    baseline_file = ['icmp_rtt.pkl']
    current_files = ['icmp_dns_ntp_rtt.pkl', 'tcp_rtt.pkl']
    baseline = load_data_from_pickle(os.path.join(path, baseline_file[0]))
    current = RTTTable()
    for file in current_files:
        temp = load_data_from_pickle(os.path.join(path, file))
        combine_rtt_table(current, temp)    
    tcptrace_data = load_data_from_pickle('./data/tcptrace_table/alltcptrace.pkl')
    tcptrace = process_tcptrace_data(tcptrace_data) if 0 else process_min_tcptrace_data(tcptrace_data)
    rtt_analysis = RTTAnalysis(current, baseline, tcptrace)
    rtt_analysis.second_to_millisecond(rtt_analysis)

    rtt_analysis.plot_every_ipaddress()
    rtt_analysis.print_rtt_samples(folder)    
    rtt_analysis.plot_every_ipaddress_samples_number()
    rtt_analysis.calculate_data_coverage()
    rtt_analysis.plot_coverage()
    '''            
    #print(rtt_analysis.current_samples.keys())
    #print(rtt_analysis.baseline_samples.keys())
    
    # to millisecond
    for key in rtt_analysis.current_samples:
        for value in rtt_analysis.current_samples[key]:
            if float(value['rtt']) > 1000:
                print(value, end='\n')
    for key in rtt_analysis.baseline_samples:
        for value in rtt_analysis.baseline_samples[key]:
            if float(value['rtt']) > 1000:
                print(value, end='\n')
    rtt_analysis.print_result('data/rtt5/result.txt')
    # compare with tcptrace
    
    count = 0
    tcptrace_dir = './data/tcptrace_table'
    for tcptrace_rtttable in os.listdir(tcptrace_dir):
        count += 1
        if count == 2:
            break
        tcptrace_table = load_data_from_pickle(os.path.join(tcptrace_dir, tcptrace_rtttable))
        temp = rtt_analysis.compare_with_tcptrce(tcptrace_table)
        result = copy.deepcopy(temp)
        result['no'] = count
        for key, value in result.items():
            print(key)
            print(value)
            print('----------------\n')
            '''
    