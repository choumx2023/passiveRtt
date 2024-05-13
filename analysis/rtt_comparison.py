import numpy as np
import matplotlib.pyplot as plt
from ..src.RttTable import RTTTable

class RttComparison:
    def __init__(self, current_table: RTTTable, baseline_table: RTTTable):
        self.current_samples = current_table.rtt_samples
        self.baseline_samples = baseline_table.rtt_samples

    def calculate_differences(self):
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

    def plot_differences(self):
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

    def find_nearest_sample(self, current_sample, baseline_samples):
        """找到基线样本中时间戳最接近的样本"""
        current_timestamp = current_sample['timestamp']
        closest_sample = min(baseline_samples, key=lambda x: abs(x['timestamp'] - current_timestamp))
        return closest_sample
    def calculate_variance(self):
        """计算基于时间匹配的RTT差异方差"""
        differences = []
        for key in self.current_samples:
            if key in self.baseline_samples:
                for current_sample in self.current_samples[key]:
                    nearest_baseline_sample = self.find_nearest_sample(current_sample, self.baseline_samples[key])
                    rtt_diff = current_sample['rtt'] - nearest_baseline_sample['rtt']
                    differences.append(rtt_diff)
        if differences:
            return np.var(differences)
        else:
            return 0
    def analyze_differences(self):
        """分析差异，计算方差和标准差，并绘制分布图"""
        differences = self.calculate_differences()
        variance = np.var(differences)
        std_dev = np.sqrt(variance)

        # 打印统计结果
        print("Variance of RTT differences:", variance)
        print("Standard deviation of RTT differences:", std_dev)

        # 绘制直方图
        plt.hist(differences, bins=30, color='blue', alpha=0.7)
        plt.title('Distribution of RTT Differences')
        plt.xlabel('RTT Difference (ms)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    def extract_rtts(self, samples):
        """从样本中提取RTT值"""
        rtts = []
        for sample_list in samples.values():
            rtts.extend(sample['rtt'] for sample in sample_list)
        return rtts
 
    def plot_rtt_distributions(self):
        """绘制当前数据和基线数据的RTT分布图"""
        current_rtts = self.extract_rtts(self.current_samples)
        baseline_rtts = self.extract_rtts(self.baseline_samples)

        plt.figure(figsize=(12, 6))
        plt.hist(current_rtts, bins=30, alpha=0.5, label='Current RTTs', color='blue')
        plt.hist(baseline_rtts, bins=30, alpha=0.5, label='Baseline RTTs', color='red')
        plt.title('RTT Distributions: Current vs. Baseline')
        plt.xlabel('RTT (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    